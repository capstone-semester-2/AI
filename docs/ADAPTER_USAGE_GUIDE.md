# DeepSpeech2 어댑터 학습 - 실행 가이드

## 📌 요약

기존의 두 가지 학습 방식에 **어댑터 학습(Adapter Training)** 기능이 추가되었습니다.

| 방식 | 명령어 파라미터 | 설명 | 용도 |
|------|----------------|------|------|
| **일반 학습** | `train=ds2_train` | 새로운 모델을 처음부터 학습 | 새 모델 개발 |
| **파인튜닝** | `train=ds2_train` + `pretrained_model_path=...` | 기존 모델의 모든 파라미터를 업데이트하며 학습 | 모델 개선 |
| **어댑터 학습** | `train=adapter_train` + `base_model_path=...` | 기존 모델 뒤에 작은 MLP를 붙여 학습 (원본 모델은 고정) | 개인화된 서비스 ⭐️ |

---

## 🚀 3가지 실행 방식

### 1️⃣ 일반 학습 (처음부터 새로운 모델 학습)

```bash
REALDATA=$(realpath ./data)
REALTXT=$(realpath ./data/transcripts.txt)

python ./kospeech1/bin/main.py \
  model=ds2 \
  train=ds2_train \
  train.batch_size=50 \
  train.dataset_path="$REALDATA" \
  train.transcripts_path="$REALTXT"
```

**특징:**
- ✅ 기본 설정 (기존 방식과 동일)
- ✅ 배치 크기 조절 가능
- 📁 저장: `outputs/last_model_checkpoint.pt`

---

### 2️⃣ 파인튜닝 (기존 모델 기반 추가 학습)

```bash
REALDATA=$(realpath ./data)
REALTXT=$(realpath ./data/transcripts.txt)

python ./kospeech1/bin/main.py \
  model=ds2 \
  train=ds2_train \
  train.batch_size=16 \
  train.dataset_path="$REALDATA" \
  train.transcripts_path="$REALTXT" \
  train.pretrained_model_path=/home/gon-mac/local/Cap/outputs/2-model/model.pt \
  train.resume=false
```

**특징:**
- ✅ 기존 모델을 로드하고 모든 파라미터를 함께 학습
- ✅ 더 많은 데이터로 모델을 개선하고 싶을 때 사용
- 📁 저장: `outputs/last_model_checkpoint.pt`

---

### 3️⃣ 어댑터 학습 (개인화 모델) ⭐️ **NEW**

```bash
REALDATA=$(realpath ./data)
REALTXT=$(realpath ./data/transcripts.txt)
BASEMODEL=$(realpath ./outputs/2-model/model.pt)

python ./kospeech1/bin/main.py \
  model=ds2 \
  train=adapter_train \
  train.batch_size=16 \
  train.dataset_path="$REALDATA" \
  train.transcripts_path="$REALTXT" \
  train.base_model_path="$BASEMODEL" \
  train.adapter_name=user_john \
  train.adapter_save_dir=./adapters \
  train.adapter_hidden_dims=[512,256] \
  train.num_epochs=10
```

**특징:**
- ✅ **`train=adapter_train` 사용** (기존과 다름!)
- ✅ **`base_model_path`** 필수 (기존 모델 경로)
- ✅ 원본 모델은 **고정** - 절대 변경 안 됨
- ✅ **MLP 어댑터만** 학습
- 📁 저장: `./adapters/user_john_adapter.pt` (매우 작음!)

---

## 🎯 어댑터 학습 필수/선택 옵션

### 필수 옵션 ❗️

```bash
train=adapter_train                          # 반드시 adapter_train 사용
train.base_model_path=/path/to/model.pt      # 기존 모델 경로 (필수)
```

### 선택 옵션 (기본값 있음)

```bash
train.adapter_name=user_john                 # 어댑터 이름 (기본: "default")
train.adapter_save_dir=./adapters            # 저장 위치 (기본: "./adapters")
train.adapter_hidden_dims=[512,256]          # 은닉층 크기 (기본: [512,256])
train.num_epochs=10                          # 에포크 수 (기본: 10)
train.batch_size=16                          # 배치 크기 (기본: 32)
```

---

## 📋 사용자별 어댑터 학습 예제

### 예제 1: 사용자 'John'의 어댑터 학습

```bash
python ./kospeech1/bin/main.py \
  model=ds2 \
  train=adapter_train \
  train.dataset_path=./data \
  train.transcripts_path=./data/transcripts.txt \
  train.base_model_path=./outputs/model.pt \
  train.adapter_name=john \
  train.num_epochs=10
```

**결과:** `./adapters/john_adapter.pt` 생성 (매우 작은 파일)

---

### 예제 2: 사용자 'Jane'의 어댑터 학습

```bash
python ./kospeech1/bin/main.py \
  model=ds2 \
  train=adapter_train \
  train.dataset_path=./data \
  train.transcripts_path=./data/transcripts.txt \
  train.base_model_path=./outputs/model.pt \
  train.adapter_name=jane \
  train.num_epochs=10
```

**결과:** `./adapters/jane_adapter.pt` 생성

---

### 예제 3: 고급 어댑터 (3층 MLP)

```bash
python ./kospeech1/bin/main.py \
  model=ds2 \
  train=adapter_train \
  train.dataset_path=./data \
  train.transcripts_path=./data/transcripts.txt \
  train.base_model_path=./outputs/model.pt \
  train.adapter_name=advanced \
  train.adapter_hidden_dims=[512,256,128] \
  train.num_epochs=15
```

**결과:** `./adapters/advanced_adapter.pt` 생성 (더 깊은 네트워크)

---

### 예제 4: 빠른 학습 (1층 MLP)

```bash
python ./kospeech1/bin/main.py \
  model=ds2 \
  train=adapter_train \
  train.dataset_path=./data \
  train.transcripts_path=./data/transcripts.txt \
  train.base_model_path=./outputs/model.pt \
  train.adapter_name=lightweight \
  train.adapter_hidden_dims=[256] \
  train.num_epochs=5
```

**결과:** `./adapters/lightweight_adapter.pt` 생성 (가장 작음, 가장 빠름)

---

## 📊 방식 비교표

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                 │  일반 학습       │  파인튜닝        │  어댑터 학습     │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 기본 모델       │ ❌ 없음          │ ✅ 필수          │ ✅ 필수          │
│ 학습 대상       │ 🔵 전체 모델    │ 🔵 전체 모델    │ 🟢 MLP만        │
│ 파라미터 변경   │ YES (모두)       │ YES (모두)       │ NO (고정)       │
│ 저장 파일       │ model.pt         │ model.pt         │ adapter.pt       │
│ 파일 크기       │ ~100MB           │ ~100MB           │ ~1-10MB         │
│ 학습 속도       │ 느림 (시간)      │ 느림 (시간)      │ 빠름 (분)       │
│ 필요 데이터     │ 매우 많음        │ 중간 정도        │ 적음 (사용자별) │
│ GPU 메모리      │ 많이 필요        │ 많이 필요        │ 적음            │
│ 사용 케이스     │ 새 모델 개발     │ 모델 개선        │ 개인화 ⭐️     │
│ 명령어 모드     │ train=ds2_train  │ train=ds2_train  │ train=adapter.. │
│                 │                  │ +pretrained_..   │ _train          │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## 📁 학습 완료 후 파일 구조

```
kospeech1/
├── outputs/
│   └── 2-model/
│       └── model.pt              ← 기본 모델 (변경 없음!)
│
├── adapters/                      ← 어댑터 저장 디렉토리
│   ├── john_adapter.pt           ← John의 어댑터 (~2MB)
│   ├── jane_adapter.pt           ← Jane의 어댑터 (~2MB)
│   ├── advanced_adapter.pt       ← 3층 어댑터 (~3MB)
│   └── lightweight_adapter.pt    ← 1층 어댑터 (~1MB)
```

**각 어댑터는:**
- 독립적으로 저장됨
- 원본 모델과 분리됨
- 사용자별로 관리 가능
- 쉽게 로드/배포 가능

---

## 🔄 전체 워크플로우

### Step 1: 기본 모델 학습 (일회)

```bash
# 일반 학습으로 새 모델 생성
python ./kospeech1/bin/main.py \
  model=ds2 \
  train=ds2_train \
  train.dataset_path=./data \
  train.transcripts_path=./data/transcripts.txt

# 결과: ./outputs/last_model_checkpoint.pt
```

### Step 2: 사용자별 어댑터 학습 (반복)

```bash
# John의 어댑터 학습
python ./kospeech1/bin/main.py \
  model=ds2 \
  train=adapter_train \
  train.dataset_path=./data \
  train.transcripts_path=./data/transcripts.txt \
  train.base_model_path=./outputs/last_model_checkpoint.pt \
  train.adapter_name=john

# Jane의 어댑터 학습
python ./kospeech1/bin/main.py \
  model=ds2 \
  train=adapter_train \
  train.dataset_path=./data \
  train.transcripts_path=./data/transcripts.txt \
  train.base_model_path=./outputs/last_model_checkpoint.pt \
  train.adapter_name=jane

# ... 더 많은 사용자의 어댑터 학습
```

### Step 3: 추론 (다음 단계)

추론 코드에서 어댑터를 로드하여 사용:

```python
from kospeech.models import AdapterManager

# 모델 로드
model = load_model('./outputs/last_model_checkpoint.pt')

# 어댑터 로드
manager = AdapterManager()
manager.load_adapter(model, './adapters/john_adapter.pt')

# 추론 실행
predictions = model(audio, lengths)
```

---

## ✅ 주요 특징 정리

### 어댑터 학습의 장점

| 기능 | 설명 |
|------|------|
| **빠른 학습** | MLP만 학습하므로 매우 빠름 |
| **적은 데이터** | 사용자별 소규모 데이터셋으로도 가능 |
| **작은 파일** | 어댑터만 저장 (~1-10MB vs 100MB) |
| **원본 보호** | 기본 모델은 절대 변경 안 됨 |
| **개인화** | 각 사용자의 음성 특성에 맞춤 |
| **효율적** | GPU 메모리 사용량 적음 |

### 예상 성능

| 항목 | 일반 학습 | 파인튜닝 | 어댑터 학습 |
|------|---------|--------|----------|
| 학습 시간 | **12시간** | **10시간** | **30분** |
| 필요 데이터 | 많음 | 중간 | 적음 |
| 파일 크기 | 100MB | 100MB | 5MB |
| 메모리 사용 | 높음 | 높음 | 낮음 |
| 개인화 | ❌ | ❌ | ✅ |

---

## 🛠️ 문제 해결

### Q: "adapter_train이 뭐예요?"
**A:** `train=adapter_train`은 어댑터 전용 학습 설정입니다. 기존의 `ds2_train` 대신 사용하세요.

### Q: "base_model_path가 없으면?"
**A:** 필수 옵션입니다. 반드시 기존 모델 경로를 지정해야 합니다.

### Q: "adapter_name은?"
**A:** 어댑터를 식별하는 이름입니다. 저장 파일명에 포함됩니다: `{adapter_name}_adapter.pt`

### Q: "adapter_hidden_dims는?"
**A:** MLP의 은닉층 크기입니다. `[512,256]`은 2층, `[512,256,128]`은 3층입니다.

### Q: "어댑터를 로드하려면?"
**A:** 다음 스크립트를 보세요:
```python
from kospeech.models import AdapterManager
manager = AdapterManager()
manager.load_adapter(model, './adapters/john_adapter.pt')
```

---

## 📚 참고 문서

- **ADAPTER_README.md**: 어댑터 API 상세 문서
- **EXECUTION_GUIDE.sh**: 실행 명령어 스크립트
- **adapter_training_example.py**: 예제 및 비교표

---

## 🎉 완성!

이제 다음 3가지 방식으로 학습할 수 있습니다:

1. ✅ **일반 학습** - 새로운 모델 개발
2. ✅ **파인튜닝** - 기존 모델 개선  
3. ✅ **어댑터 학습** - 개인화된 서비스 ⭐️

각 방식의 명령어를 구별하여 사용하세요! 🚀
