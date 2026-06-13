"""FastAPI Real-ESRGAN upscale server with job/poll pattern.

Endpoints:
  POST   /jobs              → {job_id, status: "queued"}
  GET    /jobs/{id}         → {status, progress?, error?}
  GET    /jobs/{id}/result  → image/jpeg binary
  DELETE /jobs/{id}         → cancel (best-effort)
  GET    /health            → {ok, device, queue_depth}

Security:
  - Bearer token from env CARDJENG_TOKEN (required)
  - Max upload 5 MB (config.MAX_UPLOAD_BYTES)
  - Max input dimension 3000 px
  - One active inference; queue up to 4; 429 when full

CORS:
  - Allowed origin: https://cardjeng.pages.dev + http://localhost:* (dev)

Run:
  CARDJENG_TOKEN=somesecret \
  ~/cardjeng-tools/venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
"""
import asyncio, io, os, time, uuid, secrets, sys, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import inference module
sys.path.insert(0, str(Path(__file__).parent))
from inference import upscale_tiled, load_model, get_device, pick_tile

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("cardjeng-upscale")

# ─── Config ─────────────────────────────────────────────────────────
TOKEN = os.environ.get("CARDJENG_TOKEN", "")
if not TOKEN:
    log.warning("⚠️ CARDJENG_TOKEN empty — auth DISABLED (personal solo use mode)")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024     # 5 MB
MAX_INPUT_EDGE = 3000                  # px
MAX_QUEUE = 4                          # waiting jobs
JOB_TTL = 600                          # seconds before stale job cleanup
# Rate limit anonymous use to avoid abuse if URL leaks
ANON_MAX_JOBS_PER_HOUR = 60            # per IP (best-effort, behind tunnel)

# ─── In-memory job store ────────────────────────────────────────────
@dataclass
class Job:
    id: str
    status: str = "queued"      # queued | running | done | failed | cancelled
    progress: float = 0.0       # 0..1
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    input_bytes: int = 0
    input_dims: Optional[tuple] = None
    output_dims: Optional[tuple] = None
    result_bytes: Optional[bytes] = None    # JPEG-encoded output
    cancelled: bool = False

jobs: dict[str, Job] = {}
queue: asyncio.Queue = asyncio.Queue()
inference_lock = asyncio.Lock()

# Simple per-IP rate limit (window: last hour)
_rate_log: dict[str, list[float]] = {}

def check_rate(ip: str):
    now = time.time()
    window = [t for t in _rate_log.get(ip, []) if now - t < 3600]
    if len(window) >= ANON_MAX_JOBS_PER_HOUR:
        raise HTTPException(429, f"Rate limit: {ANON_MAX_JOBS_PER_HOUR}/hour per IP")
    window.append(now)
    _rate_log[ip] = window

# ─── Auth ───────────────────────────────────────────────────────────
def check_auth(authorization: str = Header(default="")):
    if not TOKEN:
        return  # disabled — relying on URL obscurity + rate limit
    expected = f"Bearer {TOKEN}"
    if not secrets.compare_digest(authorization, expected):
        raise HTTPException(401, "Invalid or missing token")

# ─── App ────────────────────────────────────────────────────────────
app = FastAPI(title="CardJeng Upscale Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cardjeng.pages.dev",
    ],
    # Plus regex for local dev origins: localhost, 127.0.0.1, 192.168.x.x, 10.x.x.x
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    expose_headers=["Content-Type", "Content-Length"],
)

# ─── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "ok": True,
        "device": get_device(),
        "queue_depth": queue.qsize(),
        "active": inference_lock.locked(),
        "jobs": len(jobs),
    }

@app.post("/jobs", dependencies=[Depends(check_auth)])
async def create_job(file: UploadFile = File(...),
                     x_forwarded_for: str = Header(default="")):
    # Best-effort IP for rate limit (Cloudflare passes original IP)
    client_ip = x_forwarded_for.split(",")[0].strip() or "anon"
    check_rate(client_ip)

    # Read with size limit
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES} bytes")
    if not data:
        raise HTTPException(400, "Empty upload")

    # Validate image + dims
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    h, w = img.shape[:2]
    if max(h, w) > MAX_INPUT_EDGE:
        raise HTTPException(413, f"Input edge {max(h, w)} exceeds max {MAX_INPUT_EDGE}")

    # Queue limit
    if queue.qsize() >= MAX_QUEUE:
        raise HTTPException(429, f"Queue full ({MAX_QUEUE}). Try again later.")

    # Create job
    job_id = uuid.uuid4().hex
    job = Job(id=job_id, input_bytes=len(data), input_dims=(w, h))
    jobs[job_id] = job
    await queue.put((job_id, img))
    log.info(f"queued {job_id} input={w}×{h} {len(data)/1024:.0f}KB queue={queue.qsize()}")
    return {"job_id": job_id, "status": "queued", "input_dims": [w, h]}

@app.get("/jobs/{job_id}", dependencies=[Depends(check_auth)])
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    resp = {
        "job_id": job.id,
        "status": job.status,
        "progress": round(job.progress, 3),
        "input_dims": job.input_dims,
    }
    if job.output_dims: resp["output_dims"] = job.output_dims
    if job.error:        resp["error"] = job.error
    if job.started:      resp["elapsed"] = round((job.finished or time.time()) - job.started, 2)
    return resp

@app.get("/jobs/{job_id}/result", dependencies=[Depends(check_auth)])
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(409, f"Job status is '{job.status}', not done")
    if not job.result_bytes:
        raise HTTPException(500, "Job done but no result bytes")
    return Response(content=job.result_bytes, media_type="image/jpeg")

@app.delete("/jobs/{job_id}", dependencies=[Depends(check_auth)])
async def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in ("done", "failed", "cancelled"):
        return {"job_id": job_id, "status": job.status, "note": "already terminal"}
    job.cancelled = True
    if job.status == "queued":
        job.status = "cancelled"
    return {"job_id": job_id, "status": job.status}

# ─── Worker ─────────────────────────────────────────────────────────
async def worker():
    log.info(f"worker started, device={get_device()}")
    log.info("loading model…")
    await asyncio.to_thread(load_model)
    log.info("model ready")
    while True:
        job_id, img = await queue.get()
        job = jobs.get(job_id)
        if not job or job.cancelled:
            queue.task_done(); continue
        async with inference_lock:
            try:
                job.status = "running"
                job.started = time.time()
                h, w = img.shape[:2]
                tile = pick_tile(max(h, w))
                log.info(f"running {job_id} {w}×{h} tile={tile}")

                def progress_cb(i, n):
                    job.progress = i / n

                out = await asyncio.to_thread(
                    upscale_tiled, img, tile, 16, progress_cb)
                oh, ow = out.shape[:2]
                job.output_dims = (ow, oh)

                # Encode JPEG q92
                ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError("cv2.imencode failed")
                job.result_bytes = buf.tobytes()
                job.status = "done"
                job.progress = 1.0
                job.finished = time.time()
                log.info(f"done {job_id} {ow}×{oh} {len(job.result_bytes)/1024:.0f}KB "
                         f"elapsed={job.finished - job.started:.1f}s")
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                job.finished = time.time()
                log.exception(f"failed {job_id}")
            finally:
                queue.task_done()

# ─── Periodic cleanup ───────────────────────────────────────────────
async def cleanup_loop():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        stale = [jid for jid, j in jobs.items()
                 if j.finished and (now - j.finished) > JOB_TTL]
        for jid in stale:
            del jobs[jid]
        if stale:
            log.info(f"cleaned {len(stale)} stale jobs, remaining={len(jobs)}")

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(worker())
    asyncio.create_task(cleanup_loop())
    log.info(f"server up. token={'SET' if TOKEN else 'UNSET (dev)'} "
             f"max_upload={MAX_UPLOAD_BYTES} max_edge={MAX_INPUT_EDGE}")

@app.get("/")
async def root():
    return {"service": "CardJeng Upscale Server", "endpoints": ["/health", "/jobs (POST)", "/jobs/{id} (GET)", "/jobs/{id}/result (GET)"]}
