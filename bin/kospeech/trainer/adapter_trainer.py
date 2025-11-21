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
import math
import time
import torch
import torch.nn as nn
import queue
import pandas as pd
from torch import Tensor
from typing import Tuple

from kospeech.optim import Optimizer
from kospeech.vocabs import Vocabulary
from kospeech.metrics import CharacterErrorRate
from kospeech.utils import logger
from kospeech.data import (
    MultiDataLoader,
    AudioDataLoader,
    SpectrogramDataset,
)
from kospeech.models.adapter_manager import AdapterManager


class AdapterTrainer(object):
    """
    Trainer class for fine-tuning MLP adapters on top of pre-trained DeepSpeech2 models.
    This trainer only trains the adapter while keeping the base model frozen.

    Args:
        optimizer (kospeech.optim.__init__.Optimizer): optimizer for training adapter
        criterion (torch.nn.Module): loss function
        trainset_list (list): list of training dataset
        validset (kospeech.data.data_loader.SpectrogramDataset): validation dataset
        num_workers (int): number of using cpu cores
        device (torch.device): device - 'cuda' or 'cpu'
        print_every (int): number of timesteps to print result after
        save_result_every (int): number of timesteps to save result after
        checkpoint_every (int): number of timesteps to checkpoint after
    """
    train_dict = {'loss': [], 'cer': []}
    valid_dict = {'loss': [], 'cer': []}
    train_step_result = {'loss': [], 'cer': []}
    TRAIN_RESULT_PATH = "adapter_train_result.csv"
    VALID_RESULT_PATH = "adapter_eval_result.csv"
    TRAIN_STEP_RESULT_PATH = "adapter_train_step_result.csv"

    def __init__(
            self,
            optimizer: Optimizer,
            criterion: nn.Module,
            trainset_list: list,
            validset: SpectrogramDataset,
            num_workers: int,
            device: torch.device,
            print_every: int,
            save_result_every: int,
            checkpoint_every: int,
            architecture: str = 'deepspeech2',
            vocab: Vocabulary = None,
            adapter_save_dir: str = './adapters',
    ) -> None:
        self.num_workers = num_workers
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainset_list = trainset_list
        self.validset = validset
        self.print_every = print_every
        self.save_result_every = save_result_every
        self.checkpoint_every = checkpoint_every
        self.device = device
        self.metric = CharacterErrorRate(vocab) if vocab else None
        self.architecture = architecture.lower()
        self.vocab = vocab
        self.adapter_save_dir = adapter_save_dir
        self.adapter_manager = AdapterManager()

        self.log_format = "step: {:4d}/{:4d}, loss: {:.6f}, " \
                          "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"

    def train(
        self,
        model: nn.Module,
        batch_size: int,
        epoch_time_step: int,
        num_epochs: int,
        adapter_name: str = 'default',
        resume: bool = False,
    ) -> nn.Module:
        """
        Run adapter training for a given model.

        Args:
            model (torch.nn.Module): DeepSpeech2 model with adapter
            batch_size (int): batch size for experiment
            epoch_time_step (int): number of time step for training
            num_epochs (int): number of epochs for training
            adapter_name (str): name for saving the adapter
            resume (bool): resume training with the latest checkpoint

        Returns:
            nn.Module: trained model
        """
        # Freeze base model parameters
        if isinstance(model, nn.DataParallel):
            model.module.freeze_base_model()
        else:
            model.freeze_base_model()

        logger.info('Base model parameters frozen. Only adapter will be trained.')
        logger.info(f'Adapter training start - Adapter name: {adapter_name}')

        # 🔥 best CER 트래킹용 변수
        best_valid_cer = float('inf')
        best_epoch = -1



        train_begin_time = time.time()

        for epoch in range(num_epochs):
            logger.info(f'Adapter Training Epoch {epoch} start')
            train_queue = queue.Queue(self.num_workers << 1)

            for trainset in self.trainset_list:
                trainset.shuffle()

            # Training
            train_loader = MultiDataLoader(
                self.trainset_list, train_queue, batch_size, self.num_workers, self.vocab.pad_id
            )
            train_loader.start()

            model, train_loss, train_cer = self._train_epoches(
                model=model,
                epoch=epoch,
                epoch_time_step=epoch_time_step,
                train_begin_time=train_begin_time,
                queue=train_queue,
                adapter_name=adapter_name,
            )
            train_loader.join()

            logger.info(f'Epoch {epoch} (Adapter Training) Loss {train_loss:.4f} CER {train_cer:.4f}')

            # Validation
            valid_queue = queue.Queue(self.num_workers << 1)
            valid_loader = AudioDataLoader(self.validset, valid_queue, batch_size, 0, self.vocab.pad_id)
            valid_loader.start()

            valid_cer = self._validate(model, valid_queue, adapter_name)
            valid_loader.join()

            logger.info(f'Epoch {epoch} (Adapter Validation) CER {valid_cer:.4f}')
            self._save_epoch_result(train_result=[self.train_dict, train_loss, train_cer],
                                     valid_result=[self.valid_dict, train_loss, valid_cer])
            logger.info(f'Epoch {epoch} Adapter training result saved!!')



            # 🔥 best CER 갱신 시 어댑터 체크포인트 저장
            if epoch == 0 or valid_cer < best_valid_cer:
                best_valid_cer = valid_cer
                best_epoch = epoch
                logger.info(
                    f"New best adapter CER {best_valid_cer:.4f} at epoch {best_epoch}, saving checkpoint..."
                )
                if isinstance(model, nn.DataParallel):
                    self.adapter_manager.save_adapter(
                        model.module,
                        self.adapter_save_dir,
                        f"{adapter_name}_best"
                    )
                else:
                    self.adapter_manager.save_adapter(
                        model,
                        self.adapter_save_dir,
                        f"{adapter_name}_best"
                    )


            torch.cuda.empty_cache()

        # # Save adapter after training
        # logger.info(f'Saving adapter: {adapter_name}')
        # if isinstance(model, nn.DataParallel):
        #     self.adapter_manager.save_adapter(model.module, self.adapter_save_dir, adapter_name)
        # else:
        #     self.adapter_manager.save_adapter(model, self.adapter_save_dir, adapter_name)

        # logger.info(f'Adapter training complete!')
        # return model

        # Save adapter after training (state_dict-only)
        logger.info(f'Saving adapter: {adapter_name}')
        if isinstance(model, nn.DataParallel):
            base_model = model.module
            self.adapter_manager.save_adapter(base_model, self.adapter_save_dir, adapter_name)
        else:
            base_model = model
            self.adapter_manager.save_adapter(base_model, self.adapter_save_dir, adapter_name)

        # # ✅ 전체 모델 객체(DeepSpeech2 + adapter)를 같이 저장
        # full_model_path = os.path.join(self.adapter_save_dir, f"{adapter_name}_full_model.pt")
        # torch.save(base_model, full_model_path)
        # logger.info(f'Full model with adapter saved to: {full_model_path}')

        logger.info(f'Adapter training complete!')
        return model





    def _train_epoches(
            self,
            model: nn.Module,
            epoch: int,
            epoch_time_step: int,
            train_begin_time: float,
            queue: queue.Queue,
            adapter_name: str,
    ) -> Tuple[nn.Module, float, float]:
        """
        Run one epoch of adapter training

        Args:
            model (torch.nn.Module): model with adapter
            epoch (int): current epoch number
            epoch_time_step (int): total time step in one epoch
            train_begin_time (float): time of train begin
            queue (queue.Queue): training queue
            adapter_name (str): adapter name for logging

        Returns:
            Tuple[nn.Module, float, float]: trained model, loss, and cer
        """
        cer = 1.0
        epoch_loss_total = 0.
        total_num = 0
        timestep = 0

        model.train()

        begin_time = epoch_begin_time = time.time()
        num_workers = self.num_workers

        while True:
            inputs, targets, input_lengths, target_lengths = queue.get()

            if inputs.shape[0] == 0:
                num_workers -= 1
                logger.debug(f'left train_loader: {num_workers}')

                if num_workers == 0:
                    break
                else:
                    continue

            self.optimizer.zero_grad()

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = torch.as_tensor(target_lengths).to(self.device)

            # Forward pass with adapter
            if isinstance(model, nn.DataParallel):
                model_outputs = model(inputs, input_lengths)
            else:
                model_outputs = model(inputs, input_lengths)

            # Handle adapter output
            if len(model_outputs) == 3:  # adapter is used
                _, output_lengths, adapter_outputs = model_outputs
                outputs = adapter_outputs          # (N, T, C) 또는 (B, T, C)
            else:  # no adapter
                outputs, output_lengths = model_outputs

            # ====== 여기부터 수정 포인트 ======
            # DeepSpeech2 원래 trainer(supervised)랑 동일하게 CTC 호출
            #   - CTCLoss(input, targets, input_lengths, target_lengths)
            #   - input: (T, N, C)
            loss = self.criterion(
                outputs.transpose(0, 1),   # (T, N, C)
                targets[:, 1:],            # (N, S-1)  <- 맨 앞 SOS 제거
                output_lengths,            # (N,)
                target_lengths,            # (N,)
            )

            epoch_loss_total += loss.item()
            # 원래는 input_lengths.sum() 을 쓰지만, 일단 구조 유지
            total_num += inputs.size(0)

            loss.backward()
            self.optimizer.step(model)


            # CER 계산도 supervised_trainer 스타일로 맞춰줌
            if self.metric is not None:
                # outputs: (N, T, C) -> class argmax
                y_hats = outputs.max(-1)[1]      # (N, T)
                cer = self.metric(targets[:, 1:], y_hats)
            # ====== 수정 끝 ======

            # # Forward pass with adapter
            # if isinstance(model, nn.DataParallel):
            #     model_outputs = model(inputs, input_lengths)
            # else:
            #     model_outputs = model(inputs, input_lengths)

            # # Handle adapter output
            # if len(model_outputs) == 3:  # adapter is used
            #     _, output_lengths, adapter_outputs = model_outputs
            #     outputs = adapter_outputs
            # else:  # no adapter
            #     outputs, output_lengths = model_outputs

            # loss = self.criterion(
            #     outputs.contiguous().view(-1, outputs.size(-1)),
            #     targets.contiguous().view(-1),
            # )

            # epoch_loss_total += loss.item()
            # total_num += inputs.size(0)

            # loss.backward()
            # self.optimizer.step()

            # if self.metric is not None:
            #     cer = self.metric(outputs, targets)

            timestep += 1

            if timestep % self.print_every == 0:
                current_time = time.time()
                elapsed = current_time - begin_time
                epoch_elapsed = current_time - epoch_begin_time
                train_elapsed = current_time - train_begin_time

                logger.info(self.log_format.format(
                    timestep,
                    epoch_time_step,
                    epoch_loss_total / total_num,
                    cer,
                    elapsed,
                    elapsed / 60.0,
                    elapsed / 3600.0,
                    self.optimizer.get_lr(),
                ))

            if timestep % self.save_result_every == 0:
                self.train_step_result['loss'].append(epoch_loss_total / total_num)
                self.train_step_result['cer'].append(cer)

        return model, epoch_loss_total / total_num, cer

    def _validate(self, model: nn.Module, queue: queue.Queue, adapter_name: str = '') -> float:
        """
        Run validation

        Args:
            model (torch.nn.Module): model to validate
            queue (queue.Queue): validation queue
            adapter_name (str): adapter name for logging

        Returns:
            float: validation cer
        """
        cer = 0.0
        model.eval()

        with torch.no_grad():
            while True:
                inputs, targets, input_lengths, target_lengths = queue.get()

                if inputs.shape[0] == 0:
                    break

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)

                # Forward pass with adapter
                if isinstance(model, nn.DataParallel):
                    model_outputs = model(inputs, input_lengths)
                else:
                    model_outputs = model(inputs, input_lengths)

                # Handle adapter output
                if len(model_outputs) == 3:  # adapter is used
                    _, output_lengths, adapter_outputs = model_outputs
                    outputs = adapter_outputs
                else:  # no adapter
                    outputs, output_lengths = model_outputs


                if self.metric is not None:
                    y_hats = outputs.max(-1)[1]          # (N, T)
                    cer += self.metric(targets[:, 1:], y_hats)

                # if self.metric is not None:
                #     cer += self.metric(outputs, targets)

        return cer

    def _save_epoch_result(self, train_result, valid_result):
        """Save training/validation results to CSV"""
        train_dict, train_loss, train_cer = train_result
        valid_dict, _, valid_cer = valid_result

        train_dict['loss'].append(train_loss)
        train_dict['cer'].append(train_cer)
        valid_dict['loss'].append(valid_cer)
        valid_dict['cer'].append(valid_cer)

        train_df = pd.DataFrame(train_dict)
        train_df.to_csv(self.TRAIN_RESULT_PATH, index=False)

        valid_df = pd.DataFrame(valid_dict)
        valid_df.to_csv(self.VALID_RESULT_PATH, index=False)
