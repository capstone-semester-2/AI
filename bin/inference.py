# -*- coding: utf-8 -*-
import os
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.serialization import add_safe_globals
from torch.nn.parallel.data_parallel import DataParallel

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer, Jasper, DeepSpeech2, ListenAttendSpell, Conformer
)
from .tools import revise

# --------------------------
# 전역 컨텍스트(싱글톤)
# --------------------------
_ctx = None  # {"model": ..., "device": "...", "vocab": ...}

# 환경변수 기본값 (서버에서 사용)
# ENV_MODEL_PATH = os.getenv("MODEL_PATH")  # 예: /models/asr.ckpt
# ENV_VOCAB_PATH = os.getenv("VOCAB_PATH", "model/aihub_character_vocabs.csv")
# ENV_DEVICE     = os.getenv("DEVICE", "cpu")  # "cpu" | "cuda" | "cuda:0"

# 너가 실제로 모델/보카브를 둔 위치 기준
ENV_MODEL_PATH = os.getenv("MODEL_PATH", "/home/ubuntu/model/model.pt")
ENV_VOCAB_PATH = os.getenv("VOCAB_PATH", "/home/ubuntu/model/aihub_character_vocabs.csv")
ENV_DEVICE     = os.getenv("DEVICE", "cuda:0")  # 기본 cuda:0


# --------------------------
# 오디오 로드/전처리
# --------------------------
def _load_pcm16le(path: str) -> np.ndarray:
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

    feat = (feat - feat.mean()) / (np.std(feat) + 1e-12)   # CMVN
    return torch.tensor(feat, dtype=torch.float32).transpose(0, 1)  # (T, 80)

# --------------------------
# 모델 로딩/추론 함수 (새로 추가)
# --------------------------
def load_model(model_path: Optional[str] = None,
               device: Optional[str] = None,
               vocab_path: Optional[str] = None):
    """체크포인트를 로드하고 eval로 전환."""
    model_path = model_path or ENV_MODEL_PATH
    device     = device or ENV_DEVICE
    vocab_path = vocab_path or ENV_VOCAB_PATH
    if not model_path:
        raise ValueError("MODEL_PATH 환경변수 또는 load_model(model_path=...)를 지정하세요.")

    # PyTorch 2.6 대응: 안전목록 + weights_only=False
    add_safe_globals([DataParallel])
    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    model = obj.module if hasattr(obj, "module") else obj
    model = model.to(device).eval()

    vocab = KsponSpeechVocabulary(vocab_path)
    return model, device, vocab

def run_inference(model, wav_path: str, device: Optional[str],
                  vocab: KsponSpeechVocabulary) -> Dict[str, Any]:
    """오디오 파일 1개에 대해 텍스트를 반환."""
    device = device or ENV_DEVICE

    # 특징 추출
    feature = parse_audio(wav_path, del_silence=False)      # (T, 80)
    input_length = torch.LongTensor([feature.size(0)])

    # 디바이스 이동
    feature = feature.to(device)

    with torch.no_grad():
        if isinstance(model, ListenAttendSpell):
            model.encoder.device = device
            model.decoder.device = device
            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, DeepSpeech2):
            model.device = device
            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, (SpeechTransformer, Jasper, Conformer)):
            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        else:
            y_hats = model.recognize(feature.unsqueeze(0), input_length)

    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
    sentence = revise(sentence)
    text = sentence[0] if isinstance(sentence, (list, tuple)) else sentence
    text = text.strip()

    return {
        "title": Path(wav_path).name,
        "text": text,
        "model_type": type(model).__name__,
    }

def get_model():
    """앱 기동시 1회 로딩 후 재사용(싱글톤). 서버에서 사용."""
    global _ctx
    if _ctx is None:
        model, device, vocab = load_model()
        _ctx = {"model": model, "device": device, "vocab": vocab}
    return _ctx

def infer_on_file(wav_path: str) -> Dict[str, Any]:
    """server.py에서 호출: 파일 경로만 주면 결과 JSON 반환."""
    ctx = get_model()
    return run_inference(ctx["model"], wav_path, ctx["device"], ctx["vocab"])

# --------------------------
# CLI 호환(기존 스크립트 동작 유지)
# --------------------------
def _cli_main():
    parser = argparse.ArgumentParser(description="KoSpeech Inference (CLI)")
    parser.add_argument("--model_path", type=str, required=not bool(ENV_MODEL_PATH))
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=ENV_DEVICE)
    parser.add_argument("--vocab_path", type=str, default=ENV_VOCAB_PATH)
    parser.add_argument("--out_json", type=str, default=None)
    opt = parser.parse_args()

    model, device, vocab = load_model(
        model_path=opt.model_path or ENV_MODEL_PATH,
        device=opt.device,
        vocab_path=opt.vocab_path,
    )
    payload = run_inference(model, opt.audio_path, device, vocab)

    out_path = Path(opt.out_json) if opt.out_json else Path(opt.audio_path).with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(payload["text"])
    print(f"[INFO] JSON saved to: {out_path}")

if __name__ == "__main__":
    _cli_main()
