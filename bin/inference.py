"""
python kospeech1/bin/inference.py \
  --multi_model_paths normal.pt hearing.pt neuro.pt \
  --model_names normal hearing neuro \
  --paths \
    "normal /path/a.wav" \
    "hearing /path/b.wav" \
    "neuro:/path/c.wav"




python kospeech1/bin/inference.py \
  --multi_model_paths \
      outputs/normal/model.pt \
      outputs/hearing/model.pt \
      outputs/neuro/model.pt \
  --model_names normal hearing neuro \
  --vocab_path outputs/2-model/aihub_character_vocabs.csv \
  --device cuda:0 \
  --warmup

  
  python kospeech1/bin/inference.py \
  --multi_model_paths \
      outputs/2-model/model.pt \
      outputs/2-model/model-exear.pt \
  --model_names normal hearing \
  --vocab_path outputs/2-model/aihub_character_vocabs.csv \
  --device cuda:0 \
  --warmup

  
normal data/ID-02-27-N-BJJ-02-01-F-36-KK_중복-4.wav
hearing data/ID-02-27-N-BJJ-02-01-F-36-KK_중복-4.wav

> normal /path/to/normal_case.wav
> hearing /path/to/hearing_case.wav
> neuro /path/to/neuro_case.wav
> /path/to/anything.wav   # -> 기본 모델(normal) 사용
> q



이거 여전히 동작
python kospeech1/bin/inference.py \
  --multi_model_paths \
      outputs/2-model/model.pt \
      outputs/2-model/model-ear.pt \
  --model_names normal hearing \
  --vocab_path outputs/2-model/aihub_character_vocabs.csv \
  --device cuda:0 \
  --warmup



python kospeech1/bin/inference.py \
  --multi_model_paths \
      outputs/2-model/model.pt \
      outputs/2-model/model.pt \
  --model_names normal hearing \
  --adapter_paths none outputs/2-model/kor-bjj.pt \
  --vocab_path outputs/2-model/aihub_character_vocabs.csv \
  --device cuda:0 \
  --warmup


  

> normal /path/to/normal_case.wav
> hearing /path/to/hearing_case.wav
> neuro /path/to/neuro_case.wav
> /path/to/anything.wav   # -> 기본 모델(normal) 사용
> q



"""








# -*- coding: utf-8 -*-
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple, List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.serialization import add_safe_globals
from torch.nn.parallel.data_parallel import DataParallel

# ==== KoSpeech deps ====
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer, Jasper, DeepSpeech2, ListenAttendSpell, Conformer, MLPAdapter,
)

from tools import revise


# -----------------------------
# Audio front-end
# -----------------------------
def _load_pcm16le(path: str) -> np.ndarray:
    """RAW PCM 16-bit little-endian mono를 float32 [-1,1]로 읽기 (Kspon 표준: 16kHz)."""
    data = np.fromfile(path, dtype=np.int16)     # s16le
    wav = data.astype(np.float32) / 32768.0      # [-1, 1]
    return wav


def parse_audio(audio_path: str, del_silence: bool = False,
                audio_extension: Optional[str] = None) -> Tensor:
    """
    - .pcm이면 RAW s16le(16k/mono)로 직접 로드
    - 그 외 포맷(wav 등)은 기존 load_audio 사용
    - 출력: (T, 80) fbank with CMVN
    """
    ext = (Path(audio_path).suffix.lower().lstrip(".") or "wav") if audio_extension is None else audio_extension.lower()

    if ext == "pcm":
        signal = _load_pcm16le(audio_path)                    # 16k 가정
    else:
        signal = load_audio(audio_path, del_silence, extension=ext)

    if signal is None:
        raise RuntimeError(f"Failed to load audio: {audio_path} (ext={ext})")

    feat = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type="hamming",
        sample_frequency=16000,                               # 명시적으로 16kHz
    ).transpose(0, 1).numpy()

    # CMVN
    feat = (feat - feat.mean()) / (np.std(feat) + 1e-12)
    return torch.FloatTensor(feat).transpose(0, 1)            # (T, 80)


# -----------------------------
# 단일 모델 엔진
# -----------------------------
class ASRInference:
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        warmup: bool = False,
        adapter_path: Optional[str] = None,   # ← NEW
    ):
        """
        모델/사전을 메모리에 고정 로딩.
        adapter_path 가 주어지면 DeepSpeech2 모델에 MLPAdapter 를 붙여서 사용.
        """
        self.device = device
        self.adapter_path = adapter_path
        self.adapter_loaded: bool = False

        # PyTorch 2.6+ 대응: 안전목록 + weights_only=False 로드, DataParallel 해제
        add_safe_globals([DataParallel])
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = obj.module if hasattr(obj, "module") else obj
        self.model = self.model.to(self.device).eval()

        # dtype 전환(optional)
        if dtype == "float16":
            self.model = self.model.half()
        elif dtype == "bfloat16":
            self.model = self.model.bfloat16()
        # else float32 default

        # Adapter 붙이기 (필요한 경우, DeepSpeech2 전용)
        if adapter_path:
            self._attach_adapter(adapter_path)

        # KoSpeech vocab
        self.vocab = KsponSpeechVocabulary(vocab_path)

        # 성능 관련 설정
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

        # optional warmup
        if warmup:
            self._warmup()

    def _attach_adapter(self, adapter_path: str) -> None:
        """DeepSpeech2 용 adapter .pt 를 로드해서 모델에 붙인다."""
        if not isinstance(self.model, DeepSpeech2):
            print(f"[WARN] adapter_path={adapter_path} 이 지정되었지만 모델이 DeepSpeech2 가 아니라서 무시합니다.")
            return

        try:
            # 🔥 PyTorch 2.6+ 기본 weights_only=True 때문에 실패했으니,
            #    여기서는 명시적으로 weights_only=False 로 "옛날 방식" 로더 사용
            ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[WARN] adapter 로드 실패 ({adapter_path}): {e}")
            return

        # 우리가 AdapterManager.save_adapter(...) 에서 저장한 형식:
        # {
        #   'adapter_state_dict': ...,
        #   'input_dim': int,
        #   'hidden_dims': list or ListConfig,
        #   'output_dim': int,
        #   'adapter_name': str,
        # }
        state_dict = ckpt.get("adapter_state_dict")
        input_dim = ckpt.get("input_dim")
        hidden_dims_raw = ckpt.get("hidden_dims")
        output_dim = ckpt.get("output_dim")

        if state_dict is None or input_dim is None or hidden_dims_raw is None or output_dim is None:
            print(f"[WARN] adapter 체크포인트 형식이 잘못되었습니다: {adapter_path}")
            return

        # 🔥 ListConfig 같은 것도 일반 list 로 변환
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


    def _warmup(self):
        dummy = torch.zeros(100, 80, dtype=torch.float32)
        lengths = torch.LongTensor([dummy.size(0)])
        dummy = dummy.to(self.device)
        if self._is_amp():
            dummy = self._to_amp(dummy)

        with torch.inference_mode():
            _ = self._recognize_tensor(dummy, lengths)

    def _is_amp(self) -> bool:
        return any(
            p.is_floating_point() and p.dtype in (torch.float16, torch.bfloat16)
            for p in self.model.parameters()
        )

    def _to_amp(self, x: torch.Tensor) -> torch.Tensor:
        # 모델 dtype에 맞춰 feature dtype도 맞춤
        dt = next(self.model.parameters()).dtype
        if dt == torch.float16:
            return x.half()
        if dt == torch.bfloat16:
            return x.bfloat16()
        return x

    def _recognize_tensor(self, feature: torch.Tensor, input_length: torch.LongTensor):
        m = self.model

        if isinstance(m, ListenAttendSpell):
            m.encoder.device = self.device
            m.decoder.device = self.device
            y_hats = m.recognize(feature.unsqueeze(0), input_length)

        elif isinstance(m, DeepSpeech2):
            # DeepSpeech2 + (optional) adapter
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
                # 기존 경로 그대로
                y_hats = m.recognize(feature.unsqueeze(0), input_length)

        elif isinstance(m, (SpeechTransformer, Jasper, Conformer)):
            y_hats = m.recognize(feature.unsqueeze(0), input_length)

        else:
            y_hats = m.recognize(feature.unsqueeze(0), input_length)

        return y_hats

    def infer_one(
        self,
        audio_path: str,
        save_json: bool = False,
        out_dir: Optional[str] = None,
        audio_extension: Optional[str] = None,
    ):
        """
        단일 파일 추론 + 시간 측정.
        return: dict(payload + timings)
        """
        t_total0 = time.perf_counter()

        # 1) 특징 추출
        t_feat0 = time.perf_counter()
        feature = parse_audio(audio_path, del_silence=False, audio_extension=audio_extension)
        t_feat1 = time.perf_counter()

        # 2) 길이/디바이스 이동
        input_length = torch.LongTensor([feature.size(0)])
        feature = feature.to(self.device)
        if self._is_amp():
            feature = self._to_amp(feature)

        # 3) 추론
        t_inf0 = time.perf_counter()
        with torch.inference_mode():
            y_hats = self._recognize_tensor(feature, input_length)
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
        t_inf1 = time.perf_counter()

        # 4) 후처리
        sentence = self.vocab.label_to_string(y_hats.cpu().detach().numpy())
        sentence = revise(sentence)
        text = sentence[0] if isinstance(sentence, (list, tuple)) else sentence
        text = text.strip()

        payload = {
            "title": Path(audio_path).name,
            "text": text,
        }

        # 5) JSON 저장 (옵션)
        if save_json:
            if out_dir:
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / (Path(audio_path).stem + ".json")
            else:
                out_path = Path(audio_path).with_suffix(".json")
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        t_total1 = time.perf_counter()

        timings_ms = {
            "feature_ms": int((t_feat1 - t_feat0) * 1000),
            "inference_ms": int((t_inf1 - t_inf0) * 1000),
            "total_ms": int((t_total1 - t_total0) * 1000),
        }
        return payload, timings_ms


# -----------------------------
# 다중 모델 엔진
# -----------------------------
class MultiASRInference:
    """
    최대 3개 모델까지 동시에 올려두고,
    이름으로 선택해서 추론하는 래퍼.
    각 모델마다 adapter 를 별도로 붙일 수 있음.
    """
    def __init__(
        self,
        model_paths: Sequence[str],
        model_names: Sequence[str],
        vocab_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        warmup: bool = False,
        adapter_paths: Optional[Sequence[Optional[str]]] = None,  # ← NEW
    ):
        if len(model_paths) == 0:
            raise ValueError("model_paths 가 비었습니다.")
        if len(model_paths) > 3:
            raise ValueError("최대 3개 모델까지만 지원합니다.")
        if len(model_names) != len(model_paths):
            raise ValueError("model_names 길이는 model_paths 길이와 같아야 합니다.")

        self.engines: Dict[str, ASRInference] = {}
        for idx, (name, path) in enumerate(zip(model_names, model_paths)):
            if name in self.engines:
                raise ValueError(f"중복된 모델 이름: {name}")

            adapter_path: Optional[str] = None
            if adapter_paths is not None and idx < len(adapter_paths):
                ap = adapter_paths[idx]
                if ap and str(ap).lower() not in ("none", "-"):
                    adapter_path = ap

            print(f"[INFO] load model '{name}' from {path}")
            if adapter_path:
                print(f"       -> adapter: {adapter_path}")

            self.engines[name] = ASRInference(
                model_path=path,
                vocab_path=vocab_path,
                device=device,
                dtype=dtype,
                warmup=warmup,
                adapter_path=adapter_path,
            )
        self.default_name = model_names[0]

    @property
    def model_names(self) -> List[str]:
        return list(self.engines.keys())

    def infer_one(
        self,
        model_name: Optional[str],
        audio_path: str,
        save_json: bool = False,
        out_dir: Optional[str] = None,
        audio_extension: Optional[str] = None,
    ):
        name = model_name or self.default_name
        if name not in self.engines:
            raise ValueError(f"알 수 없는 모델 이름: {name} (사용 가능: {self.model_names})")
        engine = self.engines[name]
        payload, t = engine.infer_one(
            audio_path=audio_path,
            save_json=save_json,
            out_dir=out_dir,
            audio_extension=audio_extension,
        )
        payload["model_name"] = name
        return payload, t



# -----------------------------
# CLI / REPL 유틸
# -----------------------------
def _parse_model_and_path(
    s: str,
    valid_models: Sequence[str],
    default_model: str,
) -> Tuple[str, str]:
    """
    한 줄에서 모델 이름과 경로를 파싱.
    지원 형식:
      - "path/to.wav"             -> (default_model, path)
      - "model_name path/to.wav"  -> (model_name, path)
      - "model_name:path/to.wav"  -> (model_name, path)
    """
    s = s.strip()
    if not s:
        return default_model, ""

    # "name:path" 형식 먼저 체크
    if ":" in s:
        maybe_name, rest = s.split(":", 1)
        maybe_name = maybe_name.strip()
        rest = rest.strip()
        if maybe_name in valid_models and rest:
            return maybe_name, rest

    # 공백 기준: "name path"
    tokens = s.split()
    if len(tokens) >= 2 and tokens[0] in valid_models:
        name = tokens[0]
        path = " ".join(tokens[1:])
        return name, path

    # 그 외: 전부 path 로 보고 default 모델 사용
    return default_model, s


def process_paths_single(engine: ASRInference, paths: Sequence[str],
                         save_json: bool, out_dir: Optional[str]):
    for p in paths:
        p = p.strip()
        if not p:
            continue
        if not Path(p).exists():
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            continue

        payload, t = engine.infer_one(p, save_json=save_json, out_dir=out_dir)
        print(f"\n=== {payload['title']} ===")
        print(payload["text"])
        print(f"[timing] feature: {t['feature_ms']} ms | inference: {t['inference_ms']} ms | total: {t['total_ms']} ms")


def process_paths_multi(multi: MultiASRInference, paths: Sequence[str],
                        save_json: bool, out_dir: Optional[str]):
    for s in paths:
        s = s.strip()
        if not s:
            continue
        model_name, path = _parse_model_and_path(s, multi.model_names, multi.default_name)
        if not path:
            continue
        if not Path(path).exists():
            print(f"[ERROR][{model_name}] File not found: {path}", file=sys.stderr)
            continue

        payload, t = multi.infer_one(model_name, path, save_json=save_json, out_dir=out_dir)
        print(f"\n=== [{payload['model_name']}] {payload['title']} ===")
        print(payload["text"])
        print(f"[timing] feature: {t['feature_ms']} ms | inference: {t['inference_ms']} ms | total: {t['total_ms']} ms")


def repl_single(engine: ASRInference, save_json: bool, out_dir: Optional[str]):
    print("-> 파일 경로를 입력하세요. 종료하려면 빈 줄 또는 q.")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line.lower() in {"q", "quit", "exit"} or line == "":
            break
        process_paths_single(engine, [line], save_json, out_dir)


def repl_multi(multi: MultiASRInference, save_json: bool, out_dir: Optional[str]):
    names = ", ".join(multi.model_names)
    print("-> 모델 이름과 파일 경로를 입력하세요. 종료하려면 빈 줄 또는 q.")
    print(f"   예) normal /path/to.wav  또는  neuro:/path/to.wav")
    print(f"   모델 이름을 생략하면 기본 모델({multi.default_name})이 사용됩니다.")
    print(f"   사용 가능 모델: {names}")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line.lower() in {"q", "quit", "exit"} or line == "":
            break
        model_name, path = _parse_model_and_path(line, multi.model_names, multi.default_name)
        process_paths_multi(multi, [f"{model_name} {path}"], save_json, out_dir)


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="KoSpeech ASR - One-time load, multi-file / multi-model inference"
    )
    # 단일 모델 모드 (기존)
    parser.add_argument("--model_path", type=str, help="단일 모델 경로")

    # 멀티 모델 모드 (최대 3개)
    parser.add_argument(
        "--multi_model_paths",
        type=str,
        nargs="+",
        help="여러 모델 경로 (최대 3개). 예: --multi_model_paths normal.pt hearing.pt neuro.pt",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="*",
        help="각 모델의 이름. 예: --model_names normal hearing neuro (미지정시 m1,m2,... 사용)",
    )
    parser.add_argument(
        "--default_model",
        type=str,
        default=None,
        help="모델 이름을 생략했을 때 사용할 기본 모델 이름 (기본: 첫 번째 모델)",
    )

    parser.add_argument(
        "--adapter_paths",
        type=str,
        nargs="*",
        help=(
            "멀티 모델 모드에서 각 모델에 대응되는 adapter .pt 경로 목록. "
            "길이가 --multi_model_paths 와 같거나 더 짧을 수 있습니다. "
            "비어있거나 'none' / '-' 인 항목은 해당 모델에서 adapter 를 사용하지 않습니다."
        ),
    )


    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="단일 모델 모드에서 사용할 adapter .pt 경로 (선택, DeepSpeech2 전용)",
    )



    parser.add_argument("--vocab_path", type=str, default="data/vocab/aihub_character_vocabs.csv")
    parser.add_argument("--device", type=str, default="cpu")  # cpu / cuda / cuda:0
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--warmup", action="store_true",
                        help="모델 로드 직후 짧은 워밍업 실행")

    # 입출력 모드
    parser.add_argument("--paths", type=str, nargs="*", help="미리 지정된 오디오 파일 리스트")
    parser.add_argument("--stdin", action="store_true", help="표준입력으로 파일경로 라인별 처리")
    parser.add_argument("--save_json", action="store_true", help="각 파일 결과를 JSON으로 저장")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="JSON 저장 디렉토리 (미지정시 입력파일 옆에 .json)")

    args = parser.parse_args()

    # 모드 결정: 멀티 모델 우선
    if args.multi_model_paths:
        if args.model_path:
            print("[WARN] --multi_model_paths 와 --model_path 가 동시에 지정되었습니다. "
                  "--multi_model_paths 를 우선 사용합니다.", file=sys.stderr)

        if len(args.multi_model_paths) > 3:
            parser.error("최대 3개 모델까지만 지원합니다 (--multi_model_paths).")

        if args.model_names:
            if len(args.model_names) != len(args.multi_model_paths):
                parser.error("--model_names 길이는 --multi_model_paths 와 같아야 합니다.")
            model_names = args.model_names
        else:
            # 기본 이름: m1, m2, ...
            model_names = [f"m{i+1}" for i in range(len(args.multi_model_paths))]

        # adapter_paths 정규화 (선택)
        adapter_paths: Optional[List[Optional[str]]] = None
        if args.adapter_paths:
            if len(args.adapter_paths) > len(args.multi_model_paths):
                parser.error("--adapter_paths 길이는 --multi_model_paths 보다 길 수 없습니다.")
            adapter_paths = list(args.adapter_paths)
            # 짧으면 뒤를 None 으로 채움
            if len(adapter_paths) < len(args.multi_model_paths):
                adapter_paths += [None] * (len(args.multi_model_paths) - len(adapter_paths))



        multi = MultiASRInference(
            model_paths=args.multi_model_paths,
            model_names=model_names,
            vocab_path=args.vocab_path,
            device=args.device,
            dtype=args.dtype,
            warmup=args.warmup,
            adapter_paths=adapter_paths,   # ← NEW
        )

        # default_model 지정
        if args.default_model:
            if args.default_model not in multi.model_names:
                parser.error(f"--default_model {args.default_model} 은/는 존재하지 않는 모델 이름입니다. "
                             f"사용 가능: {multi.model_names}")
            multi.default_name = args.default_model

        if args.paths:
            process_paths_multi(multi, args.paths, args.save_json, args.out_dir)
        elif args.stdin:
            lines = [line.rstrip("\n") for line in sys.stdin]
            process_paths_multi(multi, lines, args.save_json, args.out_dir)
        else:
            repl_multi(multi, args.save_json, args.out_dir)

    else:
        # 단일 모델 모드 (이전과 동일)
        if not args.model_path:
            parser.error("단일 모델 모드에서는 --model_path 를 지정해야 합니다 "
                         "(또는 멀티 모델 모드로 --multi_model_paths 사용).")

        engine = ASRInference(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            device=args.device,
            dtype=args.dtype,
            warmup=args.warmup,
            adapter_path=args.adapter_path,
        )

        if args.paths:
            process_paths_single(engine, args.paths, args.save_json, args.out_dir)
        elif args.stdin:
            lines = [line.rstrip("\n") for line in sys.stdin]
            process_paths_single(engine, lines, args.save_json, args.out_dir)
        else:
            repl_single(engine, args.save_json, args.out_dir)


if __name__ == "__main__":
    main()
