# -- 상단 import 부 --
import os, tempfile, hashlib, time, asyncio, re, logging, signal
from typing import Optional, Dict, Any
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
IDLE_SHUTDOWN_S = int(os.getenv("IDLE_SHUTDOWN_S", 60000))  # ★ 유휴 종료 임계
HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

# 프로덕션: S3만
# SAFE_HOST = re.compile(r"(^|\.)amazonaws\.com$", re.IGNORECASE)
# 로컬 테스트용:
# SAFE_HOST = re.compile(r"(localhost|127\.0\.0\.1|(^|\.)amazonaws\.com$)", re.IGNORECASE)

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
    get_model()  # 웜업
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


@app.post("/api/korean")
async def korean(req: AnalyzeReq):
    t0 = time.time()

    print(f"Audio URL: {req.audioUrl}")

    # # 허용 도메인 체크
    # host = httpx.URL(str(req.audioUrl)).host or ""
    # if not SAFE_HOST.search(host):
    #     raise HTTPException(400, "audioUrl must be an S3 presigned URL (amazonaws.com).")

    # 사용자 단일 처리(옵션) - 이제 EmitterId 기반
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

    h = hashlib.sha256()
    tmp_path = None

    try:
        # (1) 다운로드
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as cli:
            head = await cli.head(str(req.audioUrl))
            cl = head.headers.get("Content-Length")
            if cl and int(cl) > MAX_BYTES:
                raise HTTPException(413, "File too large.")

            f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_path = f.name

            async with cli.stream("GET", str(req.audioUrl)) as r:
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

        # (2) 길이 제한 체크
        try:
            info = torchaudio.info(tmp_path)
            if info.num_frames and info.sample_rate:
                sec = info.num_frames / float(info.sample_rate)
                if sec > MAX_SECONDS:
                    raise HTTPException(413, f"Audio too long: {sec:.1f}s > {MAX_SECONDS}s")
        except Exception:
            # info 얻기 실패해도 추론은 시도
            pass

        # (3) 추론(타임아웃)
        from functools import partial
        loop = asyncio.get_running_loop()
        infer_call = partial(infer_on_file, tmp_path)

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, infer_call),
                timeout=INFER_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Inference timeout")

        # (4) 응답
        original_name = os.path.basename(urlparse(str(req.audioUrl)).path)
        result["title"] = original_name

        t2 = time.time()



        print(f"result: {result}")

        out: Dict[str, Any] = {
            "sha256": h.hexdigest(),
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
