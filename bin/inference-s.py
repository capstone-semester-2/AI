# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Optional, Dict, Any, Sequence

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.serialization import add_safe_globals
from torch.nn.parallel.data_parallel import DataParallel

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer, Jasper, DeepSpeech2, ListenAttendSpell, Conformer, MLPAdapter
)
from .tools import revise

# --------------------------
# 서버용 설정 (3 base + 1 adapter)
# --------------------------
# 👇 이 4개 파일 경로만 네 상황에 맞게 고치면 됨
ENV_MODEL_1_PATH  = os.getenv("MODEL_PATH_1",  "/home/ubuntu/model/model1.pt")
ENV_MODEL_2_PATH = os.getenv("MODEL_PATH_2", "/home/ubuntu/model/model2.pt")
ENV_MODEL_3_PATH   = os.getenv("MODEL_PATH_3",   "/home/ubuntu/model/model3.pt")

# hearing 베이스에 붙일 adapter-only 체크포인트
ENV_ADAPTER_PATH       = os.getenv("ADAPTER_PATH_2", "/home/ubuntu/model/adp.pt")

ENV_VOCAB_PATH = os.getenv("VOCAB_PATH", "/home/ubuntu/model/aihub_character_vocabs.csv")
ENV_DEVICE     = os.getenv("DEVICE", "cuda:0")  # 기본 cuda:0

# 서버에서 사용할 엔진 4개:
#  - normal    : 일반 모델
#  - hearing   : 언어청각장애
#  - neuro     : 뇌성마비
#  - adapter   : 모델 + 개인 어댑터
SERVER_MODEL_CONFIG = [
    {
        "name": "korean",
        "model_path": ENV_MODEL_1_PATH,
        "adapter_path": None,                # 어댑터 없음
    },
    {
        "name": "hearing",
        "model_path": ENV_MODEL_2_PATH,
        "adapter_path": None,                # 어댑터 없음
    },
    {
        "name": "neuro",
        "model_path": ENV_MODEL_3_PATH,
        "adapter_path": None,                # 어댑터 없음
    },
    {
        "name": "adapter",
        "model_path": ENV_MODEL_2_PATH, 
        "adapter_path": ENV_ADAPTER_PATH, 
    },
]

# --------------------------
# 오디오 로드/전처리
# --------------------------
def _load_pcm16le(path: str) -> np.ndarray:
    """RAW PCM 16-bit little-endian mono를 float32 [-1,1]로 읽기 (16kHz 가정)."""
    data = np.fromfile(path, dtype=np.int16)
    wav = data.astype(np.float32) / 32768.0
    return wav

def parse_audio(audio_path: str, del_silence: bool = False,
                audio_extension: Optional[str] = None) -> Tensor:
    """
    - .pcm이면 RAW s16le(16k/mono)로 직접 로드
    - 그 외 포맷(wav 등)은 kospeech.load_audio 사용
    - 출력: (T, 80) fbank with CMVN
    """
    ext = (Path(audio_path).suffix.lower().lstrip(".") or "wav") if audio_extension is None else audio_extension
    if ext == "pcm":
        signal = _load_pcm16le(audio_path)  # 16k 가정
    else:
        signal = load_audio(audio_path, del_silence, extension=ext)

    if signal is None:
        raise RuntimeError(f"Failed to load audio: {audio_path} (ext={ext})")

    feat = torchaudio.compliance.kaldi.fbank(
        waveform=torch.tensor(signal, dtype=torch.float32).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type="hamming",
        sample_frequency=16000,
    ).transpose(0, 1).numpy()

    # CMVN
    feat = (feat - feat.mean()) / (np.std(feat) + 1e-12)
    return torch.tensor(feat, dtype=torch.float32).transpose(0, 1)  # (T, 80)

# --------------------------
# ASR 엔진 (어댑터 지원)
# --------------------------
class ASRServerEngine:
    """
    서버에서 쓰는 단일 ASR 엔진.
    - DeepSpeech2 / Transformer / Conformer 등 지원
    - adapter_path 가 주어지면 DeepSpeech2 위에 MLPAdapter 를 붙여서 사용
    """

    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        adapter_path: Optional[str] = None,
        warmup: bool = True,
    ):
        self.device = device
        self.adapter_path = adapter_path
        self.adapter_loaded: bool = False

        # PyTorch 2.6 대응: 안전목록 + weights_only=False
        add_safe_globals([DataParallel])
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
        model = obj.module if hasattr(obj, "module") else obj
        self.model = model.to(self.device).eval()

        # dtype 전환(optional)
        if dtype == "float16":
            self.model = self.model.half()
        elif dtype == "bfloat16":
            self.model = self.model.bfloat16()

        # vocab
        self.vocab = KsponSpeechVocabulary(vocab_path)

        # Adapter 붙이기 (필요한 경우, DeepSpeech2 전용)
        if adapter_path:
            self._attach_adapter(adapter_path)

        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

        if warmup:
            self._warmup()

    # ---------- 어댑터 로딩 ----------
    def _attach_adapter(self, adapter_path: str) -> None:
        """DeepSpeech2 용 adapter .pt 를 로드해서 모델에 붙인다."""
        if not isinstance(self.model, DeepSpeech2):
            print(f"[WARN] adapter_path={adapter_path} 이 지정되었지만 모델이 DeepSpeech2 가 아니라서 무시합니다.")
            return

        try:
            # PyTorch 2.6+ : weights_only=True 기본이라 실패하므로 명시적으로 False
            ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[WARN] adapter 로드 실패 ({adapter_path}): {e}")
            return

        state_dict = ckpt.get("adapter_state_dict")
        input_dim = ckpt.get("input_dim")
        hidden_dims_raw = ckpt.get("hidden_dims")
        output_dim = ckpt.get("output_dim")

        if state_dict is None or input_dim is None or hidden_dims_raw is None or output_dim is None:
            print(f"[WARN] adapter 체크포인트 형식이 잘못되었습니다: {adapter_path}")
            return

        # ListConfig / tuple 등도 일반 list 로 정규화
        try:
            hidden_dims = list(hidden_dims_raw)
        except TypeError:
            hidden_dims = [int(hidden_dims_raw)]

        adapter = MLPAdapter(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_p=0.0,  # 추론에서는 dropout 안 씀
        )
        adapter.load_state_dict(state_dict)
        adapter = adapter.to(self.device)

        # 모델에 부착
        self.model.adapter = adapter
        setattr(self.model, "use_adapter", True)

        self.adapter_loaded = True
        print(f"[INFO] Adapter loaded and attached from: {adapter_path}")

    # ---------- 내부 유틸 ----------
    def _is_amp(self) -> bool:
        return any(
            p.is_floating_point() and p.dtype in (torch.float16, torch.bfloat16)
            for p in self.model.parameters()
        )

    def _to_amp(self, x: torch.Tensor) -> Tensor:
        dt = next(self.model.parameters()).dtype
        if dt == torch.float16:
            return x.half()
        if dt == torch.bfloat16:
            return x.bfloat16()
        return x

    def _warmup(self):
        dummy = torch.zeros(100, 80, dtype=torch.float32)
        lengths = torch.LongTensor([dummy.size(0)])
        dummy = dummy.to(self.device)
        if self._is_amp():
            dummy = self._to_amp(dummy)

        with torch.inference_mode():
            _ = self._recognize_tensor(dummy, lengths)

    # ---------- 실제 인퍼런스 ----------
    def _recognize_tensor(self, feature: Tensor, input_length: torch.LongTensor):
        m = self.model

        if isinstance(m, ListenAttendSpell):
            m.encoder.device = self.device
            m.decoder.device = self.device
            y_hats = m.recognize(feature.unsqueeze(0), input_length)

        elif isinstance(m, DeepSpeech2):
            m.device = self.device
            use_adapter = getattr(m, "use_adapter", False) and getattr(m, "adapter", None) is not None

            if use_adapter:
                # forward 를 직접 호출해 adapter 출력을 받아 decode
                outputs = m(feature.unsqueeze(0), input_length)
                if isinstance(outputs, (tuple, list)):
                    if len(outputs) == 3:
                        _, _, adapter_log_probs = outputs
                        predicted_log_probs = adapter_log_probs
                    elif len(outputs) == 2:
                        predicted_log_probs, _ = outputs
                    else:
                        predicted_log_probs = outputs[0]
                else:
                    predicted_log_probs = outputs

                if getattr(m, "decoder", None) is not None:
                    y_hats = m.decoder.decode(predicted_log_probs)
                else:
                    y_hats = m.decode(predicted_log_probs)
            else:
                # 기존 경로
                y_hats = m.recognize(feature.unsqueeze(0), input_length)

        elif isinstance(m, (SpeechTransformer, Jasper, Conformer)):
            y_hats = m.recognize(feature.unsqueeze(0), input_length)

        else:
            y_hats = m.recognize(feature.unsqueeze(0), input_length)

        return y_hats

    def infer_one(
        self,
        audio_path: str,
        audio_extension: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        서버용 단일 파일 추론.
        - 입력: 로컬 wav/pcm 경로
        - 출력: {"title":..., "text":..., "model_type":...}
        """
        feature = parse_audio(audio_path, del_silence=False, audio_extension=audio_extension)
        input_length = torch.LongTensor([feature.size(0)])

        feature = feature.to(self.device)
        if self._is_amp():
            feature = self._to_amp(feature)

        with torch.inference_mode():
            y_hats = self._recognize_tensor(feature, input_length)
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()

        sentence = self.vocab.label_to_string(y_hats.cpu().detach().numpy())
        sentence = revise(sentence)
        text = sentence[0] if isinstance(sentence, (list, tuple)) else sentence
        text = text.strip()

        return {
            "title": Path(audio_path).name,
            "text": text,
            "model_type": type(self.model).__name__,
        }

# --------------------------
# 전역 컨텍스트 (싱글톤)
# --------------------------
_ctx: Optional[Dict[str, Any]] = None
DEFAULT_MODEL_NAME = "normal"  # infer_on_file 에서 model_name 안 넘길 때 기본값

def _build_server_engines(
    model_config: Sequence[Dict[str, Optional[str]]],
    vocab_path: str,
    device: str,
) -> Dict[str, ASRServerEngine]:
    engines: Dict[str, ASRServerEngine] = {}
    for cfg in model_config:
        name = cfg["name"]
        model_path = cfg["model_path"]
        adapter_path = cfg.get("adapter_path")

        if not model_path:
            raise ValueError(f"model_path for '{name}' 가 비어 있습니다.")

        print(f"[SERVER] load model '{name}' from {model_path}")
        if adapter_path:
            print(f"         -> adapter: {adapter_path}")

        engines[name] = ASRServerEngine(
            model_path=model_path,
            vocab_path=vocab_path,
            device=device,
            dtype="float32",
            adapter_path=adapter_path,
            warmup=True,  # 서버 기동 시 한 번 워밍업
        )

    return engines

# 기존 서버 코드랑 호환되도록 함수 이름을 유지한다.
def get_model() -> Dict[str, Any]:
    """
    서버 기동 시 1회 호출해서 로딩만 해두는 용도.
    기존에는 단일 모델이었지만, 이제는 engines 딕셔너리를 가진 컨텍스트를 반환.
    """
    global _ctx
    if _ctx is None:
        engines = _build_server_engines(SERVER_MODEL_CONFIG, ENV_VOCAB_PATH, ENV_DEVICE)
        _ctx = {
            "engines": engines,
            "device": ENV_DEVICE,
        }
    return _ctx

def infer_on_file(wav_path: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    FastAPI 서버에서 호출하는 엔트리.
    - wav_path: 로컬에 다운로드된 wav/pcm 경로
    - model_name (옵션):
        * "normal"
        * "hearing"
        * "neuro"
        * "adapter"
      지정 안 하면 DEFAULT_MODEL_NAME 사용.
    """
    ctx = get_model()
    engines: Dict[str, ASRServerEngine] = ctx["engines"]

    name = model_name or DEFAULT_MODEL_NAME
    if name not in engines:
        raise ValueError(f"알 수 없는 모델 이름: {name}. 사용 가능: {list(engines.keys())}")

    engine = engines[name]
    result = engine.infer_one(wav_path)
    result["model_name"] = name
    return result

# 편의용 래퍼 (서버에서 직접 써도 됨)
def infer_normal(wav_path: str) -> Dict[str, Any]:
    return infer_on_file(wav_path, model_name="normal")

def infer_hearing(wav_path: str) -> Dict[str, Any]:
    return infer_on_file(wav_path, model_name="hearing")

def infer_neuro(wav_path: str) -> Dict[str, Any]:
    return infer_on_file(wav_path, model_name="neuro")

def infer_hearing_adapter(wav_path: str) -> Dict[str, Any]:
    return infer_on_file(wav_path, model_name="hearing_adapter")

# --------------------------
# (옵션) CLI 테스트용
# --------------------------
if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="KoSpeech Server Inference (3 base + 1 adapter)")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="korean",
                        help="korean | hearing | neuro | adapter")
    args = parser.parse_args()

    out = infer_on_file(args.audio_path, model_name=args.model_name)
    print(out["text"])
    print("[INFO]", _json.dumps(out, ensure_ascii=False, indent=2))
