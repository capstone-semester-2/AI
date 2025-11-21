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

        # 🔁 Post-Encoder Residual adapter
        #   - encoder 출력 h: (B, T, H)
        #   - adapter(h): (B, T, H)
        #   - h' = h + adapter(h)
        if use_adapter:
            if adapter_hidden_dims is None:
                adapter_hidden_dims = [512, 256]

            self.adapter = MLPAdapter(
                input_dim=rnn_output_size,
                hidden_dims=adapter_hidden_dims,
                output_dim=rnn_output_size,   # ⬅️ 기존 num_classes → rnn_output_size 로 변경
                dropout_p=dropout_p,
            )
            self.use_adapter = True
        else:
            self.adapter = None
            self.use_adapter = False




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
        Forward propagate inputs for CTC training.

        Returns:
            - use_adapter == False:
                (log_probs, output_lengths)
            - use_adapter == True:
                (base_log_probs, output_lengths, adapter_log_probs)
                  * base_log_probs    : 원본 encoder 출력 사용
                  * adapter_log_probs : residual adapter 적용 후 출력 사용
        """
        # 1) Encoder (conv + RNN)
        outputs, output_lengths = self.conv(inputs, input_lengths)
        outputs = outputs.permute(1, 0, 2).contiguous()

        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)

        # outputs: (T, B, H) -> (B, T, H)
        outputs = outputs.transpose(0, 1)

        # 2) 어댑터 사용 여부 (구버전 checkpoint 대비 getattr 사용)
        use_adapter = getattr(self, "use_adapter", False)
        adapter = getattr(self, "adapter", None)

        if use_adapter and adapter is not None:
            # base features (encoder output)
            base_features = outputs                        # (B, T, H)

            # residual: h' = h + adapter(h)
            residual = adapter(base_features)              # (B, T, H)
            adapted_features = base_features + residual    # (B, T, H)

            # logits -> log_probs
            base_logits = self.fc(base_features)           # (B, T, C)
            adapter_logits = self.fc(adapted_features)     # (B, T, C)

            base_log_probs = base_logits.log_softmax(dim=-1)
            adapter_log_probs = adapter_logits.log_softmax(dim=-1)

            # AdapterTrainer 에서 len(...) == 3 기준으로 처리하고 있으므로 형식 유지
            return base_log_probs, output_lengths, adapter_log_probs
        else:
            logits = self.fc(outputs)
            log_probs = logits.log_softmax(dim=-1)
            return log_probs, output_lengths



    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        CTC decoding helper.

        - use_adapter == False -> base branch 사용
        - use_adapter == True  -> adapter branch 사용
        """
        outputs = self.forward(inputs, input_lengths)

        # use_adapter=True 이면 (base_log_probs, lengths, adapter_log_probs)
        if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
            base_log_probs, output_lengths, adapter_log_probs = outputs

            if getattr(self, "use_adapter", False):
                predicted_log_probs = adapter_log_probs
            else:
                predicted_log_probs = base_log_probs
        else:
            # use_adapter=False -> (log_probs, lengths)
            predicted_log_probs, output_lengths = outputs

        return self.decode(predicted_log_probs)
