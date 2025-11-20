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

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class MLPAdapter(nn.Module):
    """
    MLP Adapter for personalized speech recognition.
    This adapter is attached to the output of the encoder and allows
    fine-tuning on user-specific data without modifying the original model.

    Args:
        input_dim (int): Input dimension (should match the RNN output dimension)
        hidden_dims (list): List of hidden layer dimensions (e.g., [512, 256, 128])
        output_dim (int): Output dimension (should match num_classes)
        dropout_p (float): Dropout probability (default: 0.1)
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dims: list,
            output_dim: int,
            dropout_p: float = 0.1,
    ):
        super(MLPAdapter, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        # Build MLP layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass of the adapter.

        Args:
            inputs (torch.FloatTensor): Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            outputs (torch.FloatTensor): Output tensor of shape (batch, seq_len, output_dim)
        """
        return self.mlp(inputs)

    def count_parameters(self) -> int:
        """Count total parameters in adapter"""
        return sum(p.numel() for p in self.parameters())

    def freeze(self) -> None:
        """Freeze adapter parameters"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze adapter parameters"""
        for param in self.parameters():
            param.requires_grad = True
