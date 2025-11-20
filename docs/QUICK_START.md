# 🎯 DeepSpeech2 MLP Adapter 구현 완료

## 📌 빠른 시작

### 어댑터 학습 실행
```bash
cd /home/gon-mac/local/Cap/kospeech1/bin

# 기본 학습 (일반)
python main.py --config-name=train train=ds2_train

# 어댑터 학습 (신규!)
python main.py --config-name=train train=adapter_train \
  train.base_model_path="path/to/pretrained_model.pt" \
  train.adapter_name="my_adapter" \
  train.adapter_save_dir="./adapters"
```

---

## 📁 생성된 파일 목록

### 1️⃣ 핵심 어댑터 모듈

#### `kospeech/models/adapter.py` (신규)
- **MLPAdapter 클래스**: 2-3층 MLP 어댑터
- 기능: forward(), freeze(), unfreeze(), count_parameters()

#### `kospeech/models/adapter_manager.py` (신규)
- **AdapterManager 클래스**: 저장/로드 관리
- 기능: save_adapter(), load_adapter(), get_adapter_info()

---

### 2️⃣ 학습 시스템

#### `kospeech/trainer/adapter_trainer.py` (신규)
- **AdapterTrainer 클래스**: 어댑터 전용 학습기
- 기능: 기본 모델 고정 + 어댑터만 학습
- 자동 저장 기능 포함

#### `kospeech/trainer/__init__.py` (수정)
- AdapterTrainer 임포트 추가
- AdapterTrainConfig 클래스 추가

---

### 3️⃣ 모델 구성

#### `kospeech/models/deepspeech2/model.py` (수정)
**추가된 파라미터:**
- `use_adapter: bool` - 어댑터 활성화 여부
- `adapter_hidden_dims: list` - 어댑터 숨겨진 차원

**추가된 메서드:**
- `freeze_base_model()` - 기본 모델 파라미터 동결
- `unfreeze_base_model()` - 기본 모델 파라미터 해제
- `count_parameters()` - 상세 파라미터 통계

**수정된 forward():**
- `use_adapter=True` 시: (base_output, output_lengths, adapter_output) 반환
- `use_adapter=False` 시: (outputs, output_lengths) 반환

#### `kospeech/model_builder.py` (수정)
- `build_deepspeech2()` 함수에 어댑터 파라미터 지원

#### `kospeech/models/__init__.py` (수정)
- MLPAdapter, AdapterManager 임포트 추가

---

### 4️⃣ 메인 학습 스크립트

#### `bin/main.py` (수정)
**추가된 함수:**
- `train_adapter()` - 어댑터 학습 전용 함수

**수정된 부분:**
- ConfigStore에 AdapterTrainConfig 등록
- main() 함수에서 자동 모드 감지 (base_model_path 확인)
- 어댑터 모드 vs 일반 학습 모드 선택

---

### 5️⃣ 문서 & 예제

#### `ADAPTER_README.md` (신규)
- 상세한 어댑터 사용 가이드
- 모든 API 문서
- 설정 파일 예제
- 트러블슈팅

#### `bin/adapter_training_example.py` (신규)
- 완전한 어댑터 학습 예제
- 함수별 설명 포함

#### `Implementation_Report_KO.md` (신규)
- 한글 구현 완료 보고서
- 아키텍처 다이어그램
- 워크플로우 설명

#### `QUICK_START.md` (이 문서)
- 빠른 시작 가이드

---

## 🎨 핵심 기능

### ✅ 1. MLP 어댑터 구조
```
입력 (RNN 출력, 1024 dim)
  ↓
선형층(1024 → 512) + ReLU + Dropout
  ↓
선형층(512 → 256) + ReLU + Dropout
  ↓
선형층(256 → 2000) + LogSoftmax
  ↓
출력 (음성 토큰)
```

### ✅ 2. 파라미터 효율성
```
전체 DeepSpeech2: ~73.5M 파라미터
기본 모델 (고정): ~66M 파라미터
어댑터 (학습): ~1.2M 파라미터

학습하는 파라미터: 1.5% 만! ⚡
```

### ✅ 3. 사용자별 어댑터 관리
```
adapters/
├── user_john_adapter.pt      # 사용자 john용
├── user_jane_adapter.pt      # 사용자 jane용
└── user_bob_adapter.pt       # 사용자 bob용
```

---

## 💻 사용 방법

### 기본 사용법

```python
from kospeech.models import DeepSpeech2, AdapterManager
from kospeech.trainer import AdapterTrainer

# 1. 어댑터가 있는 모델 생성
model = DeepSpeech2(
    input_dim=256,
    num_classes=2000,
    use_adapter=True,
    adapter_hidden_dims=[512, 256],  # 2-3층
    device=device
)

# 2. 기본 모델 파라미터 동결
model.freeze_base_model()

# 3. 어댑터 학습
trainer = AdapterTrainer(optimizer, criterion, ...)
model = trainer.train(model, batch_size=32, num_epochs=10, 
                      adapter_name='user_john')
# 자동으로 ./adapters/user_john_adapter.pt 저장됨!

# 4. 나중에 어댑터 로드
manager = AdapterManager()
manager.load_adapter(model, './adapters/user_john_adapter.pt')
```

### 명령줄 실행

```bash
# 어댑터 학습
python main.py --config-name=train train=adapter_train \
  train.base_model_path="models/pretrained_deepspeech2.pt" \
  train.adapter_name="user_john" \
  train.batch_size=32 \
  train.num_epochs=10

# 결과: ./adapters/user_john_adapter.pt 생성
```

---

## 📊 주요 클래스 요약

### MLPAdapter
```python
MLPAdapter(
    input_dim=1024,              # RNN 출력 차원
    hidden_dims=[512, 256],      # 숨겨진 레이어 차원들
    output_dim=2000,             # 출력 클래스 수
    dropout_p=0.1
)
```

### AdapterManager
```python
# 저장
AdapterManager.save_adapter(model, './adapters', 'user_john')

# 로드
AdapterManager.load_adapter(model, './adapters/user_john_adapter.pt')

# 정보 조회
info = AdapterManager.get_adapter_info('./adapters/user_john_adapter.pt')
# Returns: {'name': 'user_john', 'input_dim': 1024, ...}
```

### AdapterTrainer
```python
trainer = AdapterTrainer(
    optimizer=optimizer,
    criterion=criterion,
    trainset_list=trainsets,
    validset=validset,
    num_workers=4,
    device=device,
    vocab=vocab,
    adapter_save_dir='./adapters'  # 자동 저장 위치
)

# 학습 (자동으로 어댑터 저장)
model = trainer.train(
    model=model,
    batch_size=32,
    epoch_time_step=1000,
    num_epochs=10,
    adapter_name='user_john'
)
```

---

## 🔄 전체 워크플로우

```
┌─ 1. 기존 학습된 모델 로드
│
├─ 2. 어댑터 추가 (use_adapter=True)
│
├─ 3. 기본 모델 동결 (freeze_base_model)
│
├─ 4. 어댑터만 학습 (AdapterTrainer)
│    ├─ Forward pass: 어댑터 출력 사용
│    ├─ Backward pass: 어댑터만 업데이트
│    └─ 기본 모델: 변경 없음
│
├─ 5. 어댑터 자동 저장
│    └─ user_john_adapter.pt
│
└─ 6. 다음 사용자를 위해 새 어댑터 학습
    (같은 기본 모델 사용)
```

---

## 📖 상세 문서 위치

| 문서 | 위치 | 설명 |
|------|------|------|
| **어댑터 가이드** | `ADAPTER_README.md` | 모든 기능 & API 설명 |
| **구현 보고서** | `Implementation_Report_KO.md` | 한글 기술 문서 |
| **예제 코드** | `bin/adapter_training_example.py` | 실행 가능한 예제 |
| **빠른 시작** | `QUICK_START.md` (이 문서) | 5분 만에 시작 |

---

## 🎯 다음 단계

### 다음에 구현할 것 (추론)
- [ ] 어댑터를 로드하여 추론 실행
- [ ] 배치 추론 지원
- [ ] 결과 포스트프로세싱

### 선택사항
- [ ] 테스트 코드 작성
- [ ] 성능 벤치마크
- [ ] 추가 어댑터 구조 지원

---

## ⚠️ 주의사항

1. **반드시 기본 모델을 로드한 후 어댑터를 추가해야 합니다**
   ```python
   # ❌ 잘못됨
   model = DeepSpeech2(use_adapter=True, ...)  # 어댑터 초기화
   
   # ✅ 올바름
   model = DeepSpeech2(use_adapter=True, ...)
   model.load_state_dict(pretrained_weights)  # 그 다음 로드
   ```

2. **학습 전에 반드시 freeze_base_model()을 호출하세요**
   ```python
   model.module.freeze_base_model()  # DataParallel 사용 시
   ```

3. **어댑터 파일명은 고유해야 합니다**
   ```python
   # 서로 다른 사용자용 어댑터
   trainer.train(..., adapter_name='user_john')
   trainer.train(..., adapter_name='user_jane')
   ```

---

## 🚀 실행 예제

### Step 1: 설정 파일 준비
```yaml
# configs/adapter_train.yaml
train:
  base_model_path: "models/deepspeech2_pretrained.pt"
  adapter_name: "user_john"
  adapter_save_dir: "./adapters"
  adapter_hidden_dims: [512, 256]
  batch_size: 32
  num_epochs: 10
```

### Step 2: 학습 실행
```bash
cd bin
python main.py --config-name=train train=adapter_train
```

### Step 3: 어댑터 확인
```bash
ls -lh adapters/
# -rw-r--r-- user_john_adapter.pt  (~5MB)
```

---

## 📞 문제 해결

### Q: 어댑터가 학습되지 않는다?
**A:** `model.module.freeze_base_model()` 호출 확인
```python
# 확인 방법
for name, param in model.named_parameters():
    if 'adapter' in name:
        print(f"{name}: requires_grad={param.requires_grad}")  # True여야 함
    else:
        print(f"{name}: requires_grad={param.requires_grad}")  # False여야 함
```

### Q: 어댑터 파일이 저장되지 않는다?
**A:** `AdapterTrainer.train()` 완료 후 자동 저장됨
```python
# 또는 수동 저장
from kospeech.models import AdapterManager
AdapterManager.save_adapter(model, './adapters', 'my_adapter')
```

### Q: 여러 어댑터를 교체하며 사용하려면?
**A:** 동일한 모델에서 어댑터만 교체
```python
# 어댑터 1 로드
AdapterManager.load_adapter(model, './adapters/user_john_adapter.pt')
results1 = model(audio)

# 어댑터 2로 교체
AdapterManager.load_adapter(model, './adapters/user_jane_adapter.pt')
results2 = model(audio)
```

---

## ✅ 체크리스트

프로젝트 시작 전 확인하세요:

- [ ] `kospeech/models/adapter.py` 존재
- [ ] `kospeech/models/adapter_manager.py` 존재
- [ ] `kospeech/trainer/adapter_trainer.py` 존재
- [ ] `kospeech/models/deepspeech2/model.py` 수정됨
- [ ] `bin/main.py`에 `train_adapter()` 함수 있음
- [ ] `ADAPTER_README.md` 읽음
- [ ] 기본 모델 .pt 파일 준비됨
- [ ] 학습 데이터 준비됨

---

## 🎉 준비 완료!

이제 다음을 할 수 있습니다:

1. ✅ **어댑터 생성** - `use_adapter=True`
2. ✅ **기본 모델 보호** - `freeze_base_model()`
3. ✅ **효율적 학습** - `AdapterTrainer`로 학습
4. ✅ **저장/로드** - `AdapterManager`로 관리

**사용자별 개인화된 음성 인식 모델을 만들 준비가 되었습니다!** 🚀

---

마지막 질문? `ADAPTER_README.md` 또는 `Implementation_Report_KO.md` 참고하세요!
