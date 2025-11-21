# Copyright (c) 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.nn as nn
from typing import Optional, Dict
from kospeech.models.adapter import MLPAdapter


class AdapterManager:
    """
    Manager for saving and loading MLP adapters.
    
    This class handles:
    - Saving adapter state dict separately from the base model
    - Loading adapters into a model
    - Managing multiple adapters
    """
    
    @staticmethod
    def save_adapter(
            model: nn.Module,
            save_path: str,
            adapter_name: str = 'default',
    ) -> bool:
        """
        Save adapter state dict to a file.
        
        Args:
            model: DeepSpeech2 model with adapter
            save_path: Directory path to save adapter
            adapter_name: Name of the adapter (default: 'default')
            
        Returns:
            bool: True if saved successfully
        """
        if not hasattr(model, 'adapter') or model.adapter is None:
            print("Model does not have an adapter to save.")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Create adapter checkpoint
        checkpoint = {
            'adapter_state_dict': model.adapter.state_dict(),
            'input_dim': model.adapter.input_dim,
            'hidden_dims': model.adapter.hidden_dims,
            'output_dim': model.adapter.output_dim,
            'adapter_name': adapter_name,
        }
        
        # Save to file
        adapter_file = os.path.join(save_path, f'{adapter_name}.pt')
        torch.save(checkpoint, adapter_file)
        print(f"Adapter saved to: {adapter_file}")
        
        return True
    
    @staticmethod
    def load_adapter(
            model: nn.Module,
            adapter_path: str,
    ) -> bool:
        """
        Load adapter state dict from a file.
        
        Args:
            model: DeepSpeech2 model with adapter
            adapter_path: Path to the adapter .pt file
            
        Returns:
            bool: True if loaded successfully
        """
        if not hasattr(model, 'adapter') or model.adapter is None:
            print("Model does not have an adapter to load.")
            return False
        
        if not os.path.exists(adapter_path):
            print(f"Adapter file not found: {adapter_path}")
            return False
        
        try:
            checkpoint = torch.load(adapter_path, map_location='cpu')
            model.adapter.load_state_dict(checkpoint['adapter_state_dict'])
            print(f"Adapter loaded from: {adapter_path}")
            return True
        except Exception as e:
            print(f"Error loading adapter: {e}")
            return False
    
    @staticmethod
    def save_model_with_adapter(
            model: nn.Module,
            model_path: str,
            adapter_path: str,
            adapter_name: str = 'default',
    ) -> bool:
        """
        Save both the base model and adapter separately.
        
        Args:
            model: DeepSpeech2 model with adapter
            model_path: Path to save the base model checkpoint
            adapter_path: Directory to save the adapter
            adapter_name: Name of the adapter
            
        Returns:
            bool: True if both saved successfully
        """
        # Save base model
        base_checkpoint = {
            'model_state_dict': model.state_dict(),
            'use_adapter': model.use_adapter,
        }
        torch.save(base_checkpoint, model_path)
        print(f"Base model saved to: {model_path}")
        
        # Save adapter
        return AdapterManager.save_adapter(model, adapter_path, adapter_name)
    
    @staticmethod
    def load_model_with_adapter(
            model: nn.Module,
            model_path: str,
            adapter_path: str,
    ) -> bool:
        """
        Load both the base model and adapter.
        
        Args:
            model: DeepSpeech2 model with adapter
            model_path: Path to the base model checkpoint
            adapter_path: Path to the adapter .pt file
            
        Returns:
            bool: True if both loaded successfully
        """
        # Load base model
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Base model loaded from: {model_path}")
            except Exception as e:
                print(f"Error loading base model: {e}")
                return False
        
        # Load adapter
        return AdapterManager.load_adapter(model, adapter_path)
    
    @staticmethod
    def get_adapter_info(adapter_path: str) -> Optional[Dict]:
        """
        Get information about a saved adapter without loading it.
        
        Args:
            adapter_path: Path to the adapter .pt file
            
        Returns:
            dict: Adapter information or None if not found
        """
        if not os.path.exists(adapter_path):
            return None
        
        try:
            checkpoint = torch.load(adapter_path, map_location='cpu')
            return {
                'name': checkpoint.get('adapter_name', 'unknown'),
                'input_dim': checkpoint.get('input_dim'),
                'hidden_dims': checkpoint.get('hidden_dims'),
                'output_dim': checkpoint.get('output_dim'),
            }
        except Exception as e:
            print(f"Error reading adapter info: {e}")
            return None
