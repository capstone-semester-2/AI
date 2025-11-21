# new-mstate.py
# -*- coding: utf-8 -*-

import os
import sys
import torch

# ==== 1) kospeech 패키지를 import 가능하게 경로 추가 ====
ROOT = os.path.dirname(os.path.abspath(__file__))      # /home/gon-mac/local/Cap
KOSPEECH_ROOT = os.path.join(ROOT, "kospeech1", "bin") # /home/gon-mac/local/Cap/kospeech1/bin
if KOSPEECH_ROOT not in sys.path:
    sys.path.insert(0, KOSPEECH_ROOT)

# 실제로 패키지가 잘 보이는지 한 번 import (언피클 때 필요)
try:
    import kospeech.models  # noqa: F401
    print("[INFO] kospeech.models import OK")
except Exception as e:
    print("[WARN] failed to import kospeech.models:", e)

# ==== 2) 변환할 파일 경로 설정 ====
BASE_PATH = "outputs/2-model/model-exear.pt"          # 원래 모델 파일
OUT_PATH  = "outputs/2-model/model-exear-state.pt"    # 새로 만들 state_dict 파일


def main():
    print(f"[INFO] loading: {BASE_PATH}")

    # 🔥 PyTorch 2.6 이후 weights_only 기본값이 True라서, 여기선 False로 명시
    ckpt = torch.load(BASE_PATH, map_location="cpu", weights_only=False)

    # 1) torch.save(model, ...) 로 저장된 경우
    if isinstance(ckpt, torch.nn.DataParallel):
        print("[INFO] checkpoint is DataParallel, using .module")
        model = ckpt.module
        state_dict = model.state_dict()

    elif isinstance(ckpt, torch.nn.Module):
        print("[INFO] checkpoint is plain nn.Module")
        state_dict = ckpt.state_dict()

    # 2) torch.save(state_dict, ...) 또는 {'state_dict': ...} 형태일 때
    elif isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            print("[INFO] found 'state_dict' key in dict")
            state_dict = ckpt["state_dict"]
        else:
            print("[INFO] checkpoint already looks like a state_dict dict")
            state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(state_dict, OUT_PATH)
    print(f"[INFO] saved state_dict to: {OUT_PATH}")
    print(f"[INFO] num params (keys): {len(state_dict)}")


if __name__ == "__main__":
    main()
