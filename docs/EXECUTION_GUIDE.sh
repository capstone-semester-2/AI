#!/bin/bash

# DeepSpeech2 with MLP Adapter - Execution Examples
# 
# 3가지 학습 방식을 구별하여 실행하는 예제

echo "=========================================="
echo "DeepSpeech2 학습 & 어댑터 학습 실행 가이드"
echo "=========================================="
echo ""

# 경로 설정 (실제 경로로 수정 필요)
REALDATA=$(realpath ./data)
REALTXT=$(realpath ./data/transcripts.txt)
PRETRAINED_MODEL="/home/gon-mac/local/Cap/outputs/2-model/model.pt"

echo ""
echo "=========================================="
echo "1️⃣  일반 학습 (기존 방식 그대로)"
echo "=========================================="
echo ""
echo "새로운 모델을 처음부터 학습하는 방식"
echo ""
echo "실행 명령어:"
echo "---"
cat << 'EOF'
REALDATA=$(realpath ./data)
REALTXT=$(realpath ./data/transcripts.txt)

python ./kospeech1/bin/main.py \
  model=ds2 \
  train=ds2_train \
  train.batch_size=50 \
  train.dataset_path="$REALDATA" \
  train.transcripts_path="$REALTXT"
EOF
echo "---"
echo ""
echo "옵션 설명:"
echo "  - model=ds2: DeepSpeech2 모델 사용"
echo "  - train=ds2_train: 일반 학습 설정"
echo "  - batch_size: 배치 크기"
echo ""

echo ""
echo "=========================================="
echo "2️⃣  파인튜닝 학습 (기존 모델 기반)"
echo "=========================================="
echo ""
echo "이미 학습된 모델을 기반으로 추가 학습하는 방식"
echo "원본 모델의 모든 파라미터가 함께 업데이트됨"
echo ""
echo "실행 명령어:"
echo "---"
cat << 'EOF'
REALDATA=$(realpath ./data)
REALTXT=$(realpath ./data/transcripts.txt)

python ./kospeech1/bin/main.py \
  model=ds2 \
  train=ds2_train \
  train.batch_size=16 \
  train.dataset_path="$REALDATA" \
  train.transcripts_path="$REALTXT" \
  train.pretrained_model_path=/path/to/pretrained_model.pt \
  train.resume=false
EOF
echo "---"
echo ""
echo "옵션 설명:"
echo "  - pretrained_model_path: 기존에 학습된 모델 경로"
echo "  - resume=false: 새로 시작 (true면 마지막 체크포인트부터 이어서 학습)"
echo ""

echo ""
echo "=========================================="
echo "3️⃣  어댑터 학습 (개인화된 서비스) ⭐️ NEW"
echo "=========================================="
echo ""
echo "기존 모델의 뒤에 작은 MLP를 붙여 학습하는 방식"
echo "✅ 원본 모델의 파라미터는 고정됨 (변하지 않음)"
echo "✅ MLP 어댑터만 학습됨"
echo "✅ 어댑터는 .pt 파일로 따로 저장됨"
echo ""
echo "실행 명령어:"
echo "---"
cat << 'EOF'
REALDATA=$(realpath ./data)
REALTXT=$(realpath ./data/transcripts.txt)

python ./kospeech1/bin/main.py \
  model=ds2 \
  train=adapter_train \
  train.batch_size=16 \
  train.dataset_path="$REALDATA" \
  train.transcripts_path="$REALTXT" \
  train.base_model_path=/path/to/pretrained_model.pt \
  train.adapter_name=user_john \
  train.adapter_save_dir=./adapters \
  train.adapter_hidden_dims=[512,256] \
  train.num_epochs=10
EOF
echo "---"
echo ""
echo "필수 옵션:"
echo "  - train=adapter_train: 어댑터 학습 설정 사용 (⭐️ 중요!)"
echo "  - base_model_path: 기존에 학습된 모델 경로 (어댑터를 붙일 기반 모델)"
echo ""
echo "선택 옵션:"
echo "  - adapter_name: 어댑터 이름 (기본값: 'default')"
echo "    → 저장될 파일명: adapters/user_john_adapter.pt"
echo "  - adapter_save_dir: 어댑터 저장 디렉토리 (기본값: './adapters')"
echo "  - adapter_hidden_dims: MLP 은닉층 크기 (기본값: [512,256])"
echo "  - num_epochs: 학습 에포크 수 (기본값: 10)"
echo ""
echo "예제:"
echo "  [사용자별 어댑터 학습]"
echo ""
cat << 'EOF'
# 사용자 'john' 어댑터 학습
python ./kospeech1/bin/main.py \
  model=ds2 train=adapter_train \
  train.dataset_path=$REALDATA \
  train.transcripts_path=$REALTXT \
  train.base_model_path=$PRETRAINED_MODEL \
  train.adapter_name=john

# 사용자 'jane' 어댑터 학습
python ./kospeech1/bin/main.py \
  model=ds2 train=adapter_train \
  train.dataset_path=$REALDATA \
  train.transcripts_path=$REALTXT \
  train.base_model_path=$PRETRAINED_MODEL \
  train.adapter_name=jane

# 더 깊은 어댑터 (3층)
python ./kospeech1/bin/main.py \
  model=ds2 train=adapter_train \
  train.dataset_path=$REALDATA \
  train.transcripts_path=$REALTXT \
  train.base_model_path=$PRETRAINED_MODEL \
  train.adapter_name=advanced \
  train.adapter_hidden_dims=[512,256,128]
EOF
echo ""
echo ""

echo ""
echo "=========================================="
echo "학습 방식 비교"
echo "=========================================="
echo ""
echo "┌──────────────┬────────────────┬────────────────┬────────────────┐"
echo "│ 구분         │ 일반 학습      │ 파인튜닝       │ 어댑터 학습    │"
echo "├──────────────┼────────────────┼────────────────┼────────────────┤"
echo "│ 기본 모델    │ 새로 생성      │ 기존 모델 로드 │ 기존 모델 로드 │"
echo "│ 학습 대상    │ 전체           │ 전체           │ MLP만          │"
echo "│ 파라미터 변경│ YES            │ YES            │ NO (고정)      │"
echo "│ 저장 파일    │ model.pt       │ model.pt       │ adapter.pt     │"
echo "│ 저장 크기    │ 크다 (큼)      │ 크다 (큼)      │ 작다 (MLP)     │"
echo "│ 학습 속도    │ 느림           │ 느림           │ 빠름           │"
echo "│ 사용 데이터  │ 많음 필요      │ 중간 정도      │ 적음 OK        │"
echo "│ 용도         │ 새 모델        │ 추가 학습      │ 개인화         │"
echo "└──────────────┴────────────────┴────────────────┴────────────────┘"
echo ""

echo ""
echo "=========================================="
echo "어댑터 저장 및 로드"
echo "=========================================="
echo ""
echo "학습 후 생성되는 파일:"
echo "  adapters/"
echo "  ├── user_john_adapter.pt     (John의 어댑터)"
echo "  ├── user_jane_adapter.pt     (Jane의 어댑터)"
echo "  └── advanced_adapter.pt      (고급 어댑터)"
echo ""
echo "각 어댑터는 독립적으로 저장되므로 다양한 사용자의 어댑터를 관리할 수 있습니다!"
echo ""

echo ""
echo "=========================================="
echo "✅ 학습 완료 후 추론 (다음 단계)"
echo "=========================================="
echo ""
echo "어댑터와 함께 추론하려면 inference 코드에서:"
echo ""
echo "  from kospeech.models import AdapterManager"
echo "  "
echo "  # 어댑터 로드"
echo "  manager = AdapterManager()"
echo "  manager.load_adapter(model, './adapters/user_john_adapter.pt')"
echo "  "
echo "  # 추론 실행"
echo "  output = model(audio, lengths)"
echo ""
echo "==========================================="
echo ""
