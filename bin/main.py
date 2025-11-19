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

import os
import random
import warnings
import torch
import torch.nn as nn
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from kospeech.data.data_loader import split_dataset
from kospeech.optim import Optimizer
from kospeech.model_builder import build_model
from kospeech.utils import (
    check_envirionment,
    get_optimizer,
    get_criterion,
    logger,
    get_lr_scheduler,
)
from kospeech.vocabs import (
    KsponSpeechVocabulary,
    LibriSpeechVocabulary,
)
from kospeech.data.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
from kospeech.models import (
    DeepSpeech2Config,
    JointCTCAttentionLASConfig,
    ListenAttendSpellConfig,
    TransformerConfig,
    JointCTCAttentionTransformerConfig,
    JasperConfig,
    ConformerSmallConfig,
    ConformerMediumConfig,
    ConformerLargeConfig,
    RNNTransducerConfig,
)
from kospeech.trainer import (
    SupervisedTrainer,
    DeepSpeech2TrainConfig,
    ListenAttendSpellTrainConfig,
    TransformerTrainConfig,
    JasperTrainConfig,
    ConformerSmallTrainConfig,
    ConformerMediumTrainConfig,
    ConformerLargeTrainConfig,
    RNNTTrainConfig
)


KSPONSPEECH_VOCAB_PATH = '../../../data/vocab/kspon_sentencepiece.vocab'
KSPONSPEECH_SP_MODEL_PATH = '../../../data/vocab/kspon_sentencepiece.model'
LIBRISPEECH_VOCAB_PATH = '../../../data/vocab/tokenizer.vocab'
LIBRISPEECH_TOKENIZER_PATH = '../../../data/vocab/tokenizer.model'


# def train(config: DictConfig) -> nn.DataParallel:
#     random.seed(config.train.seed)
#     torch.manual_seed(config.train.seed)
#     torch.cuda.manual_seed_all(config.train.seed)
#     device = check_envirionment(config.train.use_cuda)
#     if hasattr(config.train, "num_threads") and int(config.train.num_threads) > 0:
#         torch.set_num_threads(config.train.num_threads)
  
#     vocab = KsponSpeechVocabulary(
#         f'../../../data/vocab/aihub_{config.train.output_unit}_vocabs.csv',
#         output_unit=config.train.output_unit,
#     )
            
#     if not config.train.resume:
#         epoch_time_step, trainset_list, validset = split_dataset(config, config.train.transcripts_path, vocab)
#         model = build_model(config, vocab, device)

#         optimizer = get_optimizer(model, config)
#         lr_scheduler = get_lr_scheduler(config, optimizer, epoch_time_step)

#         optimizer = Optimizer(optimizer, lr_scheduler, config.train.total_steps, config.train.max_grad_norm)
#         criterion = get_criterion(config, vocab)

#     else:
#         trainset_list = None
#         validset = None
#         model = None
#         optimizer = None
#         epoch_time_step = None
#         criterion = get_criterion(config, vocab)

#     trainer = SupervisedTrainer(
#         optimizer=optimizer,
#         criterion=criterion,
#         trainset_list=trainset_list,
#         validset=validset,
#         num_workers=config.train.num_workers,
#         device=device,
#         teacher_forcing_step=config.model.teacher_forcing_step,
#         min_teacher_forcing_ratio=config.model.min_teacher_forcing_ratio,
#         print_every=config.train.print_every,
#         save_result_every=config.train.save_result_every,
#         checkpoint_every=config.train.checkpoint_every,
#         architecture=config.model.architecture,
#         vocab=vocab,
#         joint_ctc_attention=config.model.joint_ctc_attention,
#     )
#     model = trainer.train(
#         model=model,
#         batch_size=config.train.batch_size,
#         epoch_time_step=epoch_time_step,
#         num_epochs=config.train.num_epochs,
#         teacher_forcing_ratio=config.model.teacher_forcing_ratio,
#         resume=config.train.resume,
#     )
#     return model



def train(config: DictConfig) -> nn.DataParallel:
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
    device = check_envirionment(config.train.use_cuda)
    if hasattr(config.train, "num_threads") and int(config.train.num_threads) > 0:
        torch.set_num_threads(config.train.num_threads)

    # aihub character vocab 그대로 사용
    vocab = KsponSpeechVocabulary(
        f'../../../data/vocab/aihub_{config.train.output_unit}_vocabs.csv',
        output_unit=config.train.output_unit,
    )

    # 새로 추가된 옵션 (config dataclass에 방금 넣어준 것)
    pretrained_path = getattr(config.train, "pretrained_model_path", "")

    if not config.train.resume:
        # 항상 새 transcripts로 데이터셋 구성
        epoch_time_step, trainset_list, validset = split_dataset(
            config, config.train.transcripts_path, vocab
        )

        # if pretrained_path:
        #     # 🔹 기존 학습된 model.pt 에서 시작 (파인튜닝)
        #     logger.info(f"Fine-tune: load pretrained model from {pretrained_path}")
        #     if config.train.use_cuda and torch.cuda.is_available():
        #         model = torch.load(pretrained_path)
        #     else:
        #         model = torch.load(pretrained_path, map_location=device)
        #         model.to(device)    
        # else:
        #     # 🔹 평소처럼 처음부터 새 모델 생성
        #     model = build_model(config, vocab, device)




        if pretrained_path:
            logger.info(f"Fine-tune: load pretrained model from {pretrained_path}")
            # 🔻 PyTorch 2.8: weights_only=False 로 옛날 방식 로드
            model = torch.load(
                pretrained_path,
                map_location=device,
                weights_only=False,
            )
        else:
            model = build_model(config, vocab, device)





        # Optimizer / scheduler 는 파인튜닝에서도 새로 생성
        optimizer = get_optimizer(model, config)
        lr_scheduler = get_lr_scheduler(config, optimizer, epoch_time_step)
        optimizer = Optimizer(
            optimizer,
            lr_scheduler,
            config.train.total_steps,
            config.train.max_grad_norm,
        )
        criterion = get_criterion(config, vocab)

    else:
        # 원래 있던 resume 로직 그대로 유지 (outputs/ 아래 checkpoint에서 이어하기)
        trainset_list = None
        validset = None
        model = None
        optimizer = None
        epoch_time_step = None
        criterion = get_criterion(config, vocab)

    trainer = SupervisedTrainer(
        optimizer=optimizer,
        criterion=criterion,
        trainset_list=trainset_list,
        validset=validset,
        num_workers=config.train.num_workers,
        device=device,
        teacher_forcing_step=config.model.teacher_forcing_step,
        min_teacher_forcing_ratio=config.model.min_teacher_forcing_ratio,
        print_every=config.train.print_every,
        save_result_every=config.train.save_result_every,
        checkpoint_every=config.train.checkpoint_every,
        architecture=config.model.architecture,
        vocab=vocab,
        joint_ctc_attention=config.model.joint_ctc_attention,
    )
    model = trainer.train(
        model=model,
        batch_size=config.train.batch_size,
        epoch_time_step=epoch_time_step,
        num_epochs=config.train.num_epochs,
        teacher_forcing_ratio=config.model.teacher_forcing_ratio,
        resume=config.train.resume,
    )
    return model





cs = ConfigStore.instance()
cs.store(group="audio", name="fbank", node=FilterBankConfig, package="audio")
cs.store(group="audio", name="melspectrogram", node=MelSpectrogramConfig, package="audio")
cs.store(group="audio", name="mfcc", node=MfccConfig, package="audio")
cs.store(group="audio", name="spectrogram", node=SpectrogramConfig, package="audio")
cs.store(group="train", name="ds2_train", node=DeepSpeech2TrainConfig, package="train")
cs.store(group="train", name="las_train", node=ListenAttendSpellTrainConfig, package="train")
cs.store(group="train", name="transformer_train", node=TransformerTrainConfig, package="train")
cs.store(group="train", name="jasper_train", node=JasperTrainConfig, package="train")
cs.store(group="train", name="conformer_small_train", node=ConformerSmallTrainConfig, package="train")
cs.store(group="train", name="conformer_medium_train", node=ConformerMediumTrainConfig, package="train")
cs.store(group="train", name="conformer_large_train", node=ConformerLargeTrainConfig, package="train")
cs.store(group="train", name="rnnt_train", node=RNNTTrainConfig, package="train")
cs.store(group="model", name="ds2", node=DeepSpeech2Config, package="model")
cs.store(group="model", name="las", node=ListenAttendSpellConfig, package="model")
cs.store(group="model", name="transformer", node=TransformerConfig, package="model")
cs.store(group="model", name="jasper", node=JasperConfig, package="model")
cs.store(group="model", name="joint-ctc-attention-las", node=JointCTCAttentionLASConfig, package="model")
cs.store(group="model", name="joint-ctc-attention-transformer", node=JointCTCAttentionTransformerConfig, package="model")
cs.store(group="model", name="conformer-small", node=ConformerSmallConfig, package="model")
cs.store(group="model", name="conformer-medium", node=ConformerMediumConfig, package="model")
cs.store(group="model", name="conformer-large", node=ConformerLargeConfig, package="model")
cs.store(group="model", name="rnnt", node=RNNTransducerConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train")
def main(config: DictConfig) -> None:
    warnings.filterwarnings('ignore')
    logger.info(OmegaConf.to_yaml(config))
    last_model_checkpoint = train(config)
    torch.save(last_model_checkpoint, os.path.join(os.getcwd(), "last_model_checkpoint.pt"))


if __name__ == '__main__':
    main()
