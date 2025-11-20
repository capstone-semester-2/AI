# Copyright (c) 2020, Soohwan Kim. All rights reserved.
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
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from kospeech.models.convolution import DeepSpeech2Extractor
from kospeech.models.model import EncoderModel
from kospeech.models.modules import Linear
from kospeech.models.adapter import MLPAdapter


class BNReluRNN(nn.Module):
    """
    Recurrent neural network with batch normalization layer & ReLU activation function.

    Args:
        input_size (int): size of input
        hidden_state_dim (int): the number of features in the hidden state `h`
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        dropout_p (float, optional): dropout probability (default: 0.1)

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs
        - **outputs**: Tensor produced by the BNReluRNN module
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_size: int,  # size of input
            hidden_state_dim: int = 512,  # dimension of RNN`s hidden state
            rnn_type: str = 'gru',  # type of RNN cell
            bidirectional: bool = True,  # if True, becomes a bidirectional rnn
            dropout_p: float = 0.1,  # dropout probability
    ):
        super(BNReluRNN, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.batch_norm = nn.BatchNorm1d(input_size)
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        total_length = inputs.size(0)

        inputs = F.relu(self.batch_norm(inputs.transpose(1, 2)))
        inputs = inputs.transpose(1, 2)

        outputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu())
        outputs, hidden_states = self.rnn(outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, total_length=total_length)

        return outputs


class DeepSpeech2(EncoderModel):
    """
    Deep Speech2 model with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1512.02595

    Args:
        input_dim (int): dimension of input vector
        num_classes (int): number of classfication
        rnn_type (str, optional): type of RNN cell (default: gru)
        num_rnn_layers (int, optional): number of recurrent layers (default: 5)
        rnn_hidden_dim (int): the number of features in the hidden state `h`
        dropout_p (float, optional): dropout probability (default: 0.1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        activation (str): type of activation function (default: hardtanh)
        device (torch.device): device - 'cuda' or 'cpu'
        use_adapter (bool, optional): whether to use MLP adapter (default: False)
        adapter_hidden_dims (list, optional): hidden dimensions for MLP adapter (default: [512, 256])

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: output
        - **output**: tensor containing the encoded features of the input sequence
    """
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            rnn_type='gru',
            num_rnn_layers: int = 5,
            rnn_hidden_dim: int = 512,
            dropout_p: float = 0.1,
            bidirectional: bool = True,
            activation: str = 'hardtanh',
            device: torch.device = 'cuda',
            use_adapter: bool = False,
            adapter_hidden_dims: Optional[list] = None,
    ):
        super(DeepSpeech2, self).__init__()
        self.device = device
        self.use_adapter = use_adapter
        self.num_classes = num_classes
        
        self.conv = DeepSpeech2Extractor(input_dim, activation=activation)
        self.rnn_layers = nn.ModuleList()
        rnn_output_size = rnn_hidden_dim << 1 if bidirectional else rnn_hidden_dim

        for idx in range(num_rnn_layers):
            self.rnn_layers.append(
                BNReluRNN(
                    input_size=self.conv.get_output_dim() if idx == 0 else rnn_output_size,
                    hidden_state_dim=rnn_hidden_dim,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    dropout_p=dropout_p,
                )
            )

        self.fc = nn.Sequential(
            nn.LayerNorm(rnn_output_size),
            Linear(rnn_output_size, num_classes, bias=False),
        )

        # Initialize MLP adapter if use_adapter is True
        if use_adapter:
            if adapter_hidden_dims is None:
                adapter_hidden_dims = [512, 256]
            
            self.adapter = MLPAdapter(
                input_dim=rnn_output_size,
                hidden_dims=adapter_hidden_dims,
                output_dim=num_classes,
                dropout_p=dropout_p,
            )
        else:
            self.adapter = None

    def freeze_base_model(self) -> None:
        """Freeze all base model parameters except adapter"""
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

    def unfreeze_base_model(self) -> None:
        """Unfreeze all base model parameters"""
        for name, param in self.named_parameters():
            param.requires_grad = True

    def count_parameters(self, trainable_only: bool = False) -> dict:
        """
        Count parameters in the model.

        Args:
            trainable_only (bool): if True, count only trainable parameters

        Returns:
            dict: dictionary containing parameter counts for different parts
        """
        total_params = 0
        adapter_params = 0
        base_params = 0

        for name, param in self.named_parameters():
            if not trainable_only or param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                
                if 'adapter' in name:
                    adapter_params += num_params
                else:
                    base_params += num_params

        return {
            'total': total_params,
            'base': base_params,
            'adapter': adapter_params,
        }

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  ctc training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor):

            * predicted_log_prob (torch.FloatTensor)s: Log probability of model predictions.
            * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``
        """
        outputs, output_lengths = self.conv(inputs, input_lengths)
        outputs = outputs.permute(1, 0, 2).contiguous()

        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)

        outputs = outputs.transpose(0, 1)  # (batch, seq_len, dim)

        # If using adapter, pass through both fc and adapter
        # 구버전 checkpoint에는 use_adapter / adapter 속성이 없을 수 있으므로 getattr 사용
        use_adapter = getattr(self, "use_adapter", False)
        adapter = getattr(self, "adapter", None)

        if use_adapter and adapter is not None:
            # Pass through fc layer (base output)
            base_outputs = self.fc(outputs).log_softmax(dim=-1)

            # Pass through adapter for personalized output
            adapter_outputs = adapter(outputs)
            adapter_outputs = adapter_outputs.log_softmax(dim=-1)

            # Return both outputs: base and adapter
            return base_outputs, output_lengths, adapter_outputs
        else:
            outputs = self.fc(outputs).log_softmax(dim=-1)
            return outputs, output_lengths


        # # If using adapter, pass through both fc and adapter
        # if self.use_adapter and self.adapter is not None:
        #     # Pass through fc layer
        #     base_outputs = self.fc(outputs).log_softmax(dim=-1)
            
        #     # Pass through adapter for personalized output
        #     adapter_outputs = self.adapter(outputs)
        #     adapter_outputs = adapter_outputs.log_softmax(dim=-1)
            
        #     # Return both outputs: base and adapter
        #     return base_outputs, output_lengths, adapter_outputs
        # else:
        #     outputs = self.fc(outputs).log_softmax(dim=-1)
        #     return outputs, output_lengths
