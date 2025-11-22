"""


{
  "EmitterId": "user-123",   // (선택)
  "pairs": [
    {
      "audioUrl": "https://.../user123_01.wav",
      "text": "나는 학교에 갑니다"
    },
    {
      "audioUrl": "https://.../user123_02.wav",
      "text": "오늘 날씨가 좋네요"
    }
  ]
}



"""





# -- 상단 import 부 --
import os, tempfile, hashlib, time, asyncio, re, logging, signal, shutil, subprocess
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import httpx
import torchaudio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, HttpUrl
import uvicorn

from kospeech1.bin.inference import get_model, infer_on_file

# ---- 설정값 ----
MAX_BYTES = int(os.getenv("MAX_BYTES", 50 * 1024 * 1024))   # 50MB
MAX_SECONDS = int(os.getenv("MAX_SECONDS", 600))            # 최대 10분 오디오
INFER_TIMEOUT_S = int(os.getenv("INFER_TIMEOUT_S", 120))    # 추론 타임아웃
CONCURRENCY = int(os.getenv("CONCURRENCY", 1))              # 동시 처리 수(=1이면 “한 번에 한 파일”)
IDLE_SHUTDOWN_S = int(os.getenv("IDLE_SHUTDOWN_S", 60000))  # 유휴 종료 임계
HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

# 어댑터 학습용 추가 설정
ADAPTER_TRAIN_TIMEOUT_S = int(os.getenv("ADAPTER_TRAIN_TIMEOUT_S", 1800))  # 학습 타임아웃 (기본 30분)

# 베이스 모델 / 어댑터 경로 (inference 모듈과 동일 env 사용)
BASE_MODEL_KOREAN = os.getenv("MODEL_PATH_1", "/home/ubuntu/model/model1.pt")
BASE_MODEL_HEARING = os.getenv("MODEL_PATH_2", "/home/ubuntu/model/model2.pt")
BASE_MODEL_NEURO = os.getenv("MODEL_PATH_3", "/home/ubuntu/model/model3.pt")

ADAPTER_PATH_HEARING = os.getenv("ADAPTER_PATH_2", "/home/ubuntu/model/adp2.pt")
ADAPTER_PATH_NEURO = os.getenv("ADAPTER_PATH_3", "/home/ubuntu/model/adp3.pt")

# adapter_train 에서 어댑터 저장 디렉토리
ADAPTER_SAVE_DIR_HEARING = os.path.dirname(ADAPTER_PATH_HEARING)
ADAPTER_SAVE_DIR_NEURO = os.path.dirname(ADAPTER_PATH_NEURO)

# adapter_train 에서 사용할 어댑터 이름 (파일명에서 확장자 제거)
ADAPTER_NAME_HEARING = os.path.splitext(os.path.basename(ADAPTER_PATH_HEARING))[0]
ADAPTER_NAME_NEURO = os.path.splitext(os.path.basename(ADAPTER_PATH_NEURO))[0]

app = FastAPI(title="AI Server")
log = logging.getLogger("ai")

# ---- 동시성 제어 ----
sem = asyncio.Semaphore(CONCURRENCY)        # 전역 동시 처리 제한
user_locks: dict[str, asyncio.Lock] = {}    # (선택) 사용자 단일 처리용


def _get_user_lock(user_id: str) -> asyncio.Lock:
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()
    return user_locks[user_id]


# ---- 유휴 종료 워처 ----
_last_req_ts = time.time()


@app.middleware("http")
async def _touch_last_request(request: Request, call_next):
    # 매 요청마다 마지막 요청 시각 갱신
    global _last_req_ts
    _last_req_ts = time.time()
    return await call_next(request)


async def _idle_watcher():
    """마지막 요청 이후 IDLE_SHUTDOWN_S 초 지나면 프로세스 종료(SIGTERM)."""
    global _last_req_ts
    try:
        while True:
            await asyncio.sleep(30)  # 30초마다 점검
            if time.time() - _last_req_ts >= IDLE_SHUTDOWN_S:
                log.info(f"[idle] no requests for {IDLE_SHUTDOWN_S}s → shutting down")
                os.kill(os.getpid(), signal.SIGTERM)  # uvicorn이 우아하게 종료
                return
    except asyncio.CancelledError:
        return


@app.on_event("startup")
async def _startup():
    # 모델 웜업
    get_model()
    app.state._idle_task = asyncio.create_task(_idle_watcher())
    log.info("model loaded (warm), idle watcher started")


@app.on_event("shutdown")
async def _shutdown():
    t = getattr(app.state, "_idle_task", None)
    if t:
        t.cancel()


@app.get("/api/health")
async def health():
    return {"ok": True}


# ==== 요청 스키마 ====

class AnalyzeReq(BaseModel):
    audioUrl: HttpUrl
    EmitterId: Optional[str] = None   # 백엔드에서 넘겨주는 식별자 (옵션)


class TrainPair(BaseModel):
    audioUrl: HttpUrl
    text: str


class AdapterTrainReq(BaseModel):
    pairs: List[TrainPair]
    EmitterId: Optional[str] = None   # 학습 요청도 동일하게 사용 가능 (락 키)


# ==== 공통 유틸 ====

async def _download_to_temp_file(url: str, suffix: str = ".wav") -> tuple[str, str, float]:
    """
    하나의 오디오 파일을 임시 경로로 다운로드.
    return: (tmp_path, sha256_hex, download_time_sec)
    """
    t0 = time.time()
    h = hashlib.sha256()
    tmp_path = None

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as cli:
        head = await cli.head(str(url))
        cl = head.headers.get("Content-Length")
        if cl and int(cl) > MAX_BYTES:
            raise HTTPException(413, "File too large.")

        f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = f.name

        async with cli.stream("GET", str(url)) as r:
            r.raise_for_status()
            n = 0
            async for chunk in r.aiter_bytes():
                n += len(chunk)
                if n > MAX_BYTES:
                    raise HTTPException(413, "File too large.")
                h.update(chunk)
                f.write(chunk)
        f.flush()

    t1 = time.time()

    # 길이 제한 체크
    try:
        info = torchaudio.info(tmp_path)
        if info.num_frames and info.sample_rate:
            sec = info.num_frames / float(info.sample_rate)
            if sec > MAX_SECONDS:
                raise HTTPException(413, f"Audio too long: {sec:.1f}s > {MAX_SECONDS}s")
    except Exception:
        # info 얻기 실패해도 추론/학습은 시도
        pass

    return tmp_path, h.hexdigest(), (t1 - t0)


async def _handle_infer(req: AnalyzeReq, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    korean / hearing / neuro / hearing-adapter / neuro-adapter 공통 처리 함수.
    """
    t0 = time.time()
    print(f"[{model_name or 'korean'}] Audio URL: {req.audioUrl}")

    # 사용자 단일 처리(옵션) - EmitterId 기반
    user_lock = _get_user_lock(req.EmitterId) if req.EmitterId else None

    # 전역 동시성 대기열
    acquired_sem = False
    try:
        await asyncio.wait_for(sem.acquire(), timeout=5)
        acquired_sem = True
    except asyncio.TimeoutError:
        raise HTTPException(429, "busy, try again later")

    if user_lock:
        await user_lock.acquire()

    tmp_path = None

    try:
        # (1) 다운로드
        tmp_path, sha256_hex, dl_sec = await _download_to_temp_file(str(req.audioUrl), suffix=".wav")
        t1 = t0 + dl_sec

        # (2) 추론(타임아웃)
        from functools import partial
        loop = asyncio.get_running_loop()
        infer_call = partial(infer_on_file, tmp_path, model_name=model_name)

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, infer_call),
                timeout=INFER_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Inference timeout")

        # (3) 응답
        original_name = os.path.basename(urlparse(str(req.audioUrl)).path)
        result["title"] = original_name

        t2 = time.time()

        print(f"[{model_name or 'korean'}] result: {result}")

        out: Dict[str, Any] = {
            "sha256": sha256_hex,
            "downloadMs": int((t1 - t0) * 1000),
            "inferenceMs": int((t2 - t1) * 1000),
            "elapsedMs": int((t2 - t0) * 1000),
            "result": result,
        }

        # EmitterId 있으면 그대로 echo
        if req.EmitterId is not None:
            out["EmitterId"] = req.EmitterId

        return out

    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if user_lock and user_lock.locked():
            user_lock.release()

        if acquired_sem:
            sem.release()


def _run_adapter_train_subprocess(
    base_model_path: str,
    adapter_save_dir: str,
    adapter_name: str,
    dataset_dir: str,
    transcripts_path: str,
    extra_override: Optional[str] = None,
):
    """
    kospeech1/bin/main.py 의 Hydra 엔트리(train=adapter_train)를 서브프로세스로 호출.
    - num_epochs 는 50 으로 고정.
    """
    cmd = [
        "python",
        "./kospeech1/bin/main.py",
        "model=ds2",
        "train=adapter_train",
        f"train.dataset_path={dataset_dir}",
        f"train.transcripts_path={transcripts_path}",
        f"train.base_model_path={base_model_path}",
        f"train.adapter_name={adapter_name}",
        "train.batch_size=3",
        "train.num_epochs=50",                 # 🔥 에포크 50 고정
        f"train.adapter_save_dir={adapter_save_dir}",
        "train.adapter_hidden_dims=[512,256]", # 필요시 조정 가능
    ]
    if extra_override:
        cmd.append(extra_override)

    print("[adapter-train] run:", " ".join(cmd))
    # 학습 실패 시 예외 발생시키도록 check=True
    subprocess.run(cmd, check=True)


async def _handle_adapter_train(
    req: AdapterTrainReq,
    base_model_path: str,
    adapter_save_dir: str,
    adapter_name: str,
) -> Dict[str, Any]:
    """
    hearing-adapter-train / neuro-adapter-train 공통 처리 함수.
    - req.pairs: [{audioUrl, text}, ...]
    """
    if not req.pairs:
        raise HTTPException(400, "pairs must not be empty")

    t0 = time.time()
    user_lock = _get_user_lock(req.EmitterId) if req.EmitterId else None

    # 전역 동시성
    acquired_sem = False
    try:
        await asyncio.wait_for(sem.acquire(), timeout=5)
        acquired_sem = True
    except asyncio.TimeoutError:
        raise HTTPException(429, "busy, try again later")

    if user_lock:
        await user_lock.acquire()

    tmp_root = None
    download_infos = []

    try:
        # (1) 임시 디렉토리 생성
        tmp_root = tempfile.mkdtemp(prefix="adapter_train_")
        audio_dir = tmp_root  # wav 들을 여기에 저장
        transcripts_path = os.path.join(tmp_root, "transcripts.txt")

        # (2) pairs 반복하며 오디오 다운로드 + transcripts 작성 준비
        # transcripts 형식: "<파일명>\t<정답 텍스트>\n"
        with open(transcripts_path, "w", encoding="utf-8") as tf:
            for idx, pair in enumerate(req.pairs):
                url = str(pair.audioUrl)
                # 파일 이름: sample_0001.wav 이런 식
                fname = f"sample_{idx+1:04d}.wav"
                local_path = os.path.join(audio_dir, fname)

                # 각 파일 별로 다운로드
                tmp_path, sha_hex, dl_sec = await _download_to_temp_file(url, suffix=".wav")
                # tmp_path -> local_path 로 이동
                shutil.move(tmp_path, local_path)

                download_infos.append({
                    "url": url,
                    "local_path": local_path,
                    "sha256": sha_hex,
                    "download_sec": dl_sec,
                })

                # transcripts 에 상대 경로(또는 파일명) 기록
                tf.write(f"{fname}\t{pair.text.strip()}\n")

        t1 = time.time()

        # (3) 서브프로세스에서 adapter_train 실행
        loop = asyncio.get_running_loop()
        train_call = lambda: _run_adapter_train_subprocess(
            base_model_path=base_model_path,
            adapter_save_dir=adapter_save_dir,
            adapter_name=adapter_name,
            dataset_dir=audio_dir,
            transcripts_path=transcripts_path,
        )

        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, train_call),
                timeout=ADAPTER_TRAIN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Adapter training timeout")
        except subprocess.CalledProcessError as e:
            # 학습 실패
            raise HTTPException(500, f"Adapter training failed: {e}") from e

        t2 = time.time()

        # (4) 응답 구성
        out: Dict[str, Any] = {
            "ok": True,
            "adapterName": adapter_name,
            "adapterSaveDir": adapter_save_dir,
            "numPairs": len(req.pairs),
            "downloadMs": int((t1 - t0) * 1000),
            "trainMs": int((t2 - t1) * 1000),
            "elapsedMs": int((t2 - t0) * 1000),
        }
        if req.EmitterId is not None:
            out["EmitterId"] = req.EmitterId

        return out

    finally:
        # 임시 디렉토리 정리
        if tmp_root and os.path.isdir(tmp_root):
            try:
                shutil.rmtree(tmp_root)
            except Exception:
                pass

        if user_lock and user_lock.locked():
            user_lock.release()

        if acquired_sem:
            sem.release()


# ==== 인퍼런스 엔드포인트 ====

@app.post("/api/korean")
async def korean(req: AnalyzeReq):
    # 기본 모델 (inference 모듈의 DEFAULT_MODEL_NAME = "korean")
    return await _handle_infer(req, model_name="korean")


@app.post("/api/hearing")
async def hearing(req: AnalyzeReq):
    # 모델2 (어댑터 없는 버전)
    return await _handle_infer(req, model_name="hearing")


@app.post("/api/neuro")
async def neuro(req: AnalyzeReq):
    # 모델3 (어댑터 없는 버전)
    return await _handle_infer(req, model_name="neuro")


@app.post("/api/hearing-adapter")
async def hearing_adapter(req: AnalyzeReq):
    # 모델2 + hearing adapter
    return await _handle_infer(req, model_name="hearing_adapter")


@app.post("/api/neuro-adapter")
async def neuro_adapter(req: AnalyzeReq):
    # 모델3 + neuro adapter
    return await _handle_infer(req, model_name="neuro_adapter")


# ==== 어댑터 학습 엔드포인트 ====

@app.post("/api/hearing-adapter-train")
async def hearing_adapter_train(req: AdapterTrainReq):
    """
    모델2(hearing 베이스)에 붙일 어댑터를 학습.
    요청 JSON 안의 pairs 에 (audioUrl + text)가 포함되어 있다.
    """
    if not BASE_MODEL_HEARING:
        raise HTTPException(500, "MODEL_PATH_2 (hearing base) is not set")

    if not ADAPTER_SAVE_DIR_HEARING:
        raise HTTPException(500, "ADAPTER_PATH_2 / ADAPTER_SAVE_DIR_HEARING not set")

    return await _handle_adapter_train(
        req=req,
        base_model_path=BASE_MODEL_HEARING,
        adapter_save_dir=ADAPTER_SAVE_DIR_HEARING,
        adapter_name=ADAPTER_NAME_HEARING,
    )


@app.post("/api/neuro-adapter-train")
async def neuro_adapter_train(req: AdapterTrainReq):
    """
    모델3(neuro 베이스)에 붙일 어댑터를 학습.
    요청 JSON 안의 pairs 에 (audioUrl + text)가 포함되어 있다.
    """
    if not BASE_MODEL_NEURO:
        raise HTTPException(500, "MODEL_PATH_3 (neuro base) is not set")

    if not ADAPTER_SAVE_DIR_NEURO:
        raise HTTPException(500, "ADAPTER_PATH_3 / ADAPTER_SAVE_DIR_NEURO not set")

    return await _handle_adapter_train(
        req=req,
        base_model_path=BASE_MODEL_NEURO,
        adapter_save_dir=ADAPTER_SAVE_DIR_NEURO,
        adapter_name=ADAPTER_NAME_NEURO,
    )


# ----(참고) uvicorn 실행 예시----
# uvicorn kospeech1.server:app --host 0.0.0.0 --port 8000
