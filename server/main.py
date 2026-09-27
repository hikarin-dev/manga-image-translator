import io
import logging
import os
import secrets
import shutil
import signal
import subprocess
import threading
import sys
from argparse import Namespace
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fastapi import FastAPI, Request, HTTPException, Header, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from pydantic import ValidationError

from manga_translator import Config
from manga_translator import stages as stage_model
from manga_translator.page_data import context_pages
from server import aux_pool
from server import capabilities
from server import edge
from server import stats
from server.instance import ExecutorInstance, executor_instances
from server.myqueue import task_queue, running_galleries, GalleryQueueElement
from server import gallery_jobs
from server.feedback import router as feedback_router
from server.feedback_review import router as feedback_review_router
from server.request_extraction import get_ctx, while_streaming, start_gallery_job, TranslateRequest, BatchTranslateRequest, get_batch_ctx
from server.to_json import to_translation, TranslationResponse

# Starlette's multipart parser rejects a body carrying more than 1000 file parts. A gallery is
# uploaded as one part per page — in a single request when the client sits on this machine — so a
# long gallery died at parse time, before any of this server's own limits had a say. Page counts
# are still bounded where that matters (see server/edge.py), and file parts spool to disk, so lift
# the parser's ceiling rather than have a second, invisible cap here. FastAPI parses the form
# before the endpoint runs and passes no arguments, so the default has to move.
_starlette_form = Request.form

def _form_without_file_cap(self, *, max_files=float('inf'), max_fields=1000, max_part_size=1024 * 1024):
    return _starlette_form(self, max_files=max_files, max_fields=max_fields, max_part_size=max_part_size)

Request.form = _form_without_file_cap

app = FastAPI()
app.include_router(feedback_router)
app.include_router(feedback_review_router)
nonce = None

BASE_DIR = Path(__file__).resolve().parent
RESULT_ROOT = (BASE_DIR.parent / "result").resolve()
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

# EdgeGate first so CORSMiddleware (added after = outermost) still stamps CORS headers on
# its rejections — the browser can't read a 401/413 body without them.
app.add_middleware(edge.EdgeGate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=edge.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加result文件夹静态文件服务
if RESULT_ROOT.exists():
    app.mount("/result", StaticFiles(directory=str(RESULT_ROOT)), name="result")

@app.post("/register", response_description="no response", tags=["internal-api"])
async def register_instance(instance: ExecutorInstance, req: Request, req_nonce: str = Header(alias="X-Nonce")):
    if req_nonce != nonce:
        raise HTTPException(401, detail="Invalid nonce")
    instance.ip = req.client.host
    executor_instances.register(instance)

@app.websocket("/aux/join")
async def aux_join(ws: WebSocket):
    """Auxiliary worker nodes dial in here and are added to the executor pool.

    Deliberately reachable from outside: an aux node is remote by definition, and EdgeGate
    only filters HTTP scopes. The join token (constant-time compared, unset by default) is
    the gate — unlike /register, no request from here can name an address we then dial."""
    await aux_pool.handle_join(ws)

@app.get("/aux/nodes", tags=["internal-api"])
async def aux_nodes() -> dict:
    """Which nodes are in the pool right now. Local-only: not in edge.PUBLIC_PATHS."""
    return {"executors": aux_pool.nodes()}

@app.get("/dashboard", response_class=HTMLResponse, tags=["ui"])
async def dashboard(req: Request) -> HTMLResponse:
    """Operator dashboard: pool state, queue, today's totals, recent jobs.

    Always available on loopback. Through the tunnel it is address-gated rather than
    token-gated (a browser navigation cannot carry X-Access-Token): reachable from
    MT_DASHBOARD_IPS and from whichever aux nodes are connected — see server.edge."""
    page = (BASE_DIR / "dashboard.html").read_text(encoding="utf-8")
    link = '<a class="btn" href="/dashboard/feedback">Feedback</a>' if edge.feedback_operator(req) else ''
    return HTMLResponse(content=page.replace('<!-- operator-feedback-link -->', link), headers={'Cache-Control': 'no-store'})

@app.get("/dashboard/data", tags=["ui"])
async def dashboard_data(req: Request) -> dict:
    """Everything the dashboard polls, in one request so its numbers are from one instant.

    Two privilege tiers. On loopback the operator sees everything, including who submitted each
    job and the source they gave. Through the tunnel — an aux node, or an allowlisted address —
    the per-job identifying fields are dropped server-side before the response is built, so the
    page can show pool health without disclosing what anyone is reading."""
    full = not bool(getattr(req.state, "external", False))
    gpu = await asyncio.to_thread(stats.gpu_snapshot)
    return {**stats.snapshot(gpu, full=full), "executors": aux_pool.nodes()}

def transform_to_image(ctx):
    # 检查是否使用占位符（在web模式下final.png保存后会设置此标记）
    if hasattr(ctx, 'use_placeholder') and ctx.use_placeholder:
        # ctx.result已经是1x1占位符图片，快速传输
        img_byte_arr = io.BytesIO()
        ctx.result.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    # 返回完整的翻译结果
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()

def transform_to_json(ctx):
    return to_translation(ctx).model_dump_json().encode("utf-8")

def transform_to_bytes(ctx):
    return to_translation(ctx).to_bytes()

def parse_config(raw: str) -> Config:
    """A client config, or a readable 400 naming the offending fields."""
    try:
        return Config.parse_raw(raw or '{}')
    except ValidationError as exc:
        problems = '; '.join(f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg')}"
                             for err in exc.errors()[:5])
        raise HTTPException(400, detail=f'invalid config: {problems}')
    except ValueError as exc:
        raise HTTPException(400, detail=f'invalid config: {exc}')


MAX_PAGE_DATA_BYTES = 8 * 1024 * 1024
AUX_FRAME_BYTES = 512 * 1024 * 1024
# The worker parameters that change stage output, as the worker this server starts sees them.
stage_runtime = stage_model.runtime_values({})


def configure_stages(args):
    """Describe the local worker's output-affecting parameters, and compute stage builds now so
    they describe the code loaded at start-up."""
    global stage_runtime
    stage_runtime = stage_model.runtime_values({
        'context_size': getattr(args, 'context_size', 0), 'pre_dict': getattr(args, 'pre_dict', None),
        'post_dict': getattr(args, 'post_dict', None)})
    threading.Thread(target=stage_model.warm, name='stage-builds', daemon=True).start()


def transform_gallery_summary(summary):
    """Final (status 0) frame of a gallery stream: the worker returns a small dict
    {count, failed}; the page images themselves arrive as status-5 frames. JSON-encode
    so the client can read which page indices failed."""
    import json
    return json.dumps(summary if isinstance(summary, dict) else {}).encode("utf-8")

@app.post("/translate/json", response_model=TranslationResponse, tags=["api", "json"],response_description="json strucure inspired by the ichigo translator extension")
async def json(req: Request, data: TranslateRequest):
    ctx = await get_ctx(req, data.config, data.image)
    return to_translation(ctx)

@app.post("/translate/bytes", response_class=StreamingResponse, tags=["api", "json"],response_description="custom byte structure for decoding look at examples in 'examples/response.*'")
async def bytes(req: Request, data: TranslateRequest):
    ctx = await get_ctx(req, data.config, data.image)
    return StreamingResponse(content=to_translation(ctx).to_bytes())

@app.post("/translate/image", response_description="the result image", tags=["api", "json"],response_class=StreamingResponse)
async def image(req: Request, data: TranslateRequest) -> StreamingResponse:
    ctx = await get_ctx(req, data.config, data.image)
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/translate/json/stream", response_class=StreamingResponse,tags=["api", "json"], response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_json(req: Request, data: TranslateRequest) -> StreamingResponse:
    return await while_streaming(req, transform_to_json, data.config, data.image)

@app.post("/translate/bytes/stream", response_class=StreamingResponse, tags=["api", "json"],response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_bytes(req: Request, data: TranslateRequest)-> StreamingResponse:
    return await while_streaming(req, transform_to_bytes,data.config, data.image)

@app.post("/translate/image/stream", response_class=StreamingResponse, tags=["api", "json"], response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_image(req: Request, data: TranslateRequest) -> StreamingResponse:
    return await while_streaming(req, transform_to_image, data.config, data.image)

@app.post("/translate/with-form/json", response_model=TranslationResponse, tags=["api", "form"],response_description="json strucure inspired by the ichigo translator extension")
async def json_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    img = await image.read()
    conf = Config.parse_raw(config)
    ctx = await get_ctx(req, conf, img)
    return to_translation(ctx)

@app.post("/translate/with-form/bytes", response_class=StreamingResponse, tags=["api", "form"],response_description="custom byte structure for decoding look at examples in 'examples/response.*'")
async def bytes_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    img = await image.read()
    conf = Config.parse_raw(config)
    ctx = await get_ctx(req, conf, img)
    return StreamingResponse(content=to_translation(ctx).to_bytes())

@app.post("/translate/with-form/image", response_description="the result image", tags=["api", "form"],response_class=StreamingResponse)
async def image_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")) -> StreamingResponse:
    img = await image.read()
    conf = Config.parse_raw(config)
    ctx = await get_ctx(req, conf, img)
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/translate/with-form/json/stream", response_class=StreamingResponse, tags=["api", "form"],response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_json_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")) -> StreamingResponse:
    img = await image.read()
    conf = Config.parse_raw(config)
    # 标记这是Web前端调用，用于占位符优化
    conf._is_web_frontend = True
    return await while_streaming(req, transform_to_json, conf, img)



@app.post("/translate/with-form/bytes/stream", response_class=StreamingResponse,tags=["api", "form"], response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_bytes_form(req: Request, image: UploadFile = File(...), config: str = Form("{}"))-> StreamingResponse:
    img = await image.read()
    conf = Config.parse_raw(config)
    return await while_streaming(req, transform_to_bytes, conf, img)

@app.post("/translate/with-form/image/stream", response_class=StreamingResponse, tags=["api", "form"], response_description="Standard streaming endpoint - returns complete image data. Suitable for API calls and scripts.")
async def stream_image_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")) -> StreamingResponse:
    """通用流式端点：返回完整图片数据，适用于API调用和comicread脚本"""
    img = await image.read()
    conf = Config.parse_raw(config)
    # 标记为通用模式，不使用占位符优化
    conf._web_frontend_optimized = False
    return await while_streaming(req, transform_to_image, conf, img)

@app.post("/translate/with-form/image/stream/web", response_class=StreamingResponse, tags=["api", "form"], response_description="Web frontend optimized streaming endpoint - uses placeholder optimization for faster response.")
async def stream_image_form_web(req: Request, image: UploadFile = File(...), config: str = Form("{}")) -> StreamingResponse:
    """Web前端专用端点：使用占位符优化，提供极速体验"""
    img = await image.read()
    conf = Config.parse_raw(config)
    # 标记为Web前端优化模式，使用占位符优化
    conf._web_frontend_optimized = True
    return await while_streaming(req, transform_to_image, conf, img)

@app.post("/translate/gallery/start", tags=["api", "form", "batch"], response_description="Create a server-owned gallery job and return immediately with its token; collect results via /translate/gallery/poll. A big gallery may arrive as several requests sharing one token (part k of n) — the job starts when the last part lands. Optional `stage` files (one per image, empty for none) carry a page's earlier pipeline data and the stage to run from (see manga_translator.page_data); `builds` is the signature of the stage builds the client planned against (409 when this server's differ); `context` gives a context-aware translator the pages before the first image, [{src, tr}] oldest first; `capture` false returns no pipeline data (no status-9 frames).")
async def start_gallery(req: Request, image: list[UploadFile] = File(...), stage: list[UploadFile] = File(None), config: str = Form("{}"), batch_size: int = Form(0), job_token: str = Form(""), part: int = Form(0), parts: int = Form(1), source_url: str = Form(""), builds: str = Form(""), context: str = Form(""), capture: bool = Form(True)) -> dict:
    req.state.benchmark = req.url.path == '/benchmark/gallery/start'
    images = [await f.read() for f in image]
    external = bool(getattr(req.state, "external", False))
    client_ip = str(getattr(req.state, "client_ip", "") or "")
    if external:
        err = edge.validate_pages(images)
        if err:
            raise HTTPException(413, detail=err)
    conf = parse_config(config)
    page_data = None
    if stage:
        if len(stage) != len(images):
            raise HTTPException(400, detail='stage files must match the images one to one')
        page_data = []
        for f in stage:
            data = await f.read(MAX_PAGE_DATA_BYTES + 1)
            if len(data) > MAX_PAGE_DATA_BYTES:
                raise HTTPException(413, detail='page data too large')
            page_data.append(data or None)
        if not any(page_data):
            page_data = None
    if builds and builds != stage_model.signature(stage_model.stage_builds(conf, stage_runtime)):
        raise HTTPException(409, detail='the translation server was updated since this translation was planned — try again')
    try:
        prior = context_pages(context) if context else None
    except ValueError as exc:
        raise HTTPException(400, detail=f'invalid context: {exc}')
    if parts > 1:
        if not (job_token and 1 < parts <= 200 and 0 <= part < parts):
            raise HTTPException(400, detail="bad part/parts")
        status, assembled, page_data = gallery_jobs.add_upload_part(
            job_token, part, parts, images, client_ip,
            max_pages=edge.MAX_PAGES_PER_JOB if external else 0, pages=page_data)
        if status == 'exists':
            return {"token": job_token, "started": True, "existing": True}
        if status == 'busy':
            raise HTTPException(503, detail="server is busy — try again in a few minutes")
        if status == 'too_many_pages':
            raise HTTPException(413, detail=f"too many pages (max {edge.MAX_PAGES_PER_JOB} per translation)")
        if status == 'pending':
            return {"token": job_token, "part": part, "received": True}
        images = assembled
    if external:
        rejected = edge.check_admission(client_ip, len(images))
        if rejected:
            raise HTTPException(rejected[0], detail=rejected[1])
    # Nothing can run this job: --delegate-only with no aux node connected, or the local worker
    # died. Refuse now with a message the client can show, rather than accepting a job that
    # would sit at 0% until the starvation guard eventually errors it.
    if executor_instances.capacity(gallery=True) == 0:
        raise HTTPException(503, detail="no translation capacity is connected right now — try again shortly")
    return await start_gallery_job(req, transform_gallery_summary, conf, images, batch_size, job_token, source_url,
                                   pages=page_data, builds=builds or None, context=prior, capture=capture)


app.add_api_route('/benchmark/gallery/start', start_gallery, methods=['POST'], tags=['benchmark'])


@app.get('/benchmark/info', tags=['benchmark'])
async def benchmark_info(req: Request):
    external = bool(getattr(req.state, 'external', False))
    s = await service_stats(req)
    return {
        'api_version': 1,
        'limits': {
            'starts_per_hour': edge.MAX_STARTS_PER_HOUR if external else None,
            'max_pages': edge.MAX_PAGES_PER_JOB if external else None,
            'max_page_bytes': edge.MAX_PAGE_BYTES if external else None,
            'max_body_bytes': edge.MAX_BODY_BYTES if external else None,
            'poll_interval_ms': 2000, 'no_client_grace_s': gallery_jobs.NO_CLIENT_GRACE_S,
        },
        'queue': s['queue'], 'workers': s['workers'], 'gpu': s['gpu'], 'uptime_s': s['uptime_s'],
        'metrics': ['stages_s', 'waits_s', 'model_loads', 'chunk_metrics', 'gpu_avg_pct',
                    'cpu_avg_pct', 'vram_max_mb', 'llm_requests', 'llm_cost_usd', 'reuse'],
    }


@app.post('/translate/gallery/resolve', tags=['api'], response_description='For a config: the effective config (fields some stage reads), the build token of each stage that applies, and the config fields each stage reads. Clients compare these with what produced their saved page data to decide which stages a run can skip. Stateless.')
async def resolve_config(config: str = Form("{}")) -> dict:
    conf = parse_config(config)
    return await asyncio.to_thread(stage_model.resolve, conf, stage_runtime)


@app.get('/capabilities', tags=['api'], response_description='What this server can run: stages, implementations (labels, versions, opaque builds, availability), options, languages and presets. Supports ETag / If-None-Match.')
async def get_capabilities(req: Request):
    body = await asyncio.to_thread(capabilities.document)
    etag = '"' + body['etag'] + '"'
    if req.headers.get('if-none-match') in (etag, body['etag']):
        return Response(status_code=304, headers={'ETag': etag})
    return JSONResponse(body, headers={'ETag': etag, 'Cache-Control': 'no-cache'})

@app.post("/translate/gallery/poll", response_class=Response, tags=["api", "batch"], response_description="Short poll. Body = a status-7 metadata frame (JSON {cursor,status,state,done,total}) + the page/study frames produced past `since` + the terminal frame once present. All in the body (not headers) so it survives cross-origin reads.")
async def poll_gallery(job_token: str = Form(...), since: int = Form(0)) -> Response:
    job = gallery_jobs.get(job_token)
    if job is None:
        # Reaped (abandoned past the grace window), evicted after finishing, or lost to a restart.
        return Response(content=gallery_jobs.GalleryJob.notfound_body(since), media_type="application/octet-stream")
    return Response(content=job.poll(since), media_type="application/octet-stream")

@app.post("/translate/gallery/cancel", tags=["api"])
async def cancel_gallery(job_token: str = Form(...)):
    """Explicitly cancel a gallery job by its client-issued token. Marks the job cancelled in
    the chunk scheduler (waiting, between chunks, or mid-chunk — no further chunks dispatch),
    cancels a still-queued chunk element, and forwards a token-scoped cancel to the worker
    when a chunk is running. Identity-correct: it can never abort a different gallery."""
    logging.getLogger('gallery-jobs').info(f'Gallery job {job_token[:8]}… cancelled by client request')
    known = gallery_jobs.cancel(job_token)
    # A job's chunks can be running on several executors at once, and others can still be
    # queued — reach every one of them, not just the first found.
    holders = running_galleries.get(job_token) or ()
    for inst in list(holders):
        await inst.cancel_gallery(job_token)
    queued = False
    for task in list(task_queue.queue):
        if isinstance(task, GalleryQueueElement) and getattr(task, 'job_token', '') == job_token:
            task.cancelled = True
            queued = True
    if queued:
        await task_queue.update_event()
    if holders or queued:
        return {"cancelling": True, "queued": queued}
    return {"cancelling": known}

app.add_api_route('/benchmark/gallery/poll', poll_gallery, methods=['POST'], tags=['benchmark'])
app.add_api_route('/benchmark/gallery/cancel', cancel_gallery, methods=['POST'], tags=['benchmark'])


@app.post("/queue-size", response_model=int, tags=["api", "json"])
async def queue_size() -> int:
    return len(task_queue.queue)

@app.get("/stats", tags=["api"])
async def service_stats(req: Request) -> dict:
    """Operator metrics: queue depth, today's jobs/pages, GPU state, recent job summaries.
    Externally reachable (token-gated by the edge middleware); per-job history persists in
    logs/jobs.jsonl.

    Only a loopback caller gets the privileged per-job fields. Holding the access token proves
    you may submit translations — not that you may read who else submitted them, or what they
    were reading."""
    gpu = await asyncio.to_thread(stats.gpu_snapshot)
    return stats.snapshot(gpu, full=not bool(getattr(req.state, "external", False)))

@app.post("/reset-context", tags=["api"])
async def reset_context():
    """Clear the worker's accumulated cross-page context. Call before translating a new
    gallery so the previous title's pages aren't used as context."""
    import aiohttp, pickle
    payload = pickle.dumps({})
    ok = 0
    for inst in executor_instances.list:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f"http://{inst.ip}:{inst.port}/simple_execute/reset_page_context", data=payload) as r:
                    if r.status == 200:
                        ok += 1
        except Exception:
            pass
    return {"ok": True, "instances_reset": ok}


@app.api_route("/result/{folder_name}/final.png", methods=["GET", "HEAD"], tags=["api", "file"])
async def get_result_by_folder(folder_name: str):
    """根据文件夹名称获取翻译结果图片"""
    result_dir = RESULT_ROOT
    if not result_dir.exists():
        raise HTTPException(404, detail="Result directory not found")

    folder_path = result_dir / folder_name
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(404, detail=f"Folder {folder_name} not found")

    final_png_path = folder_path / "final.png"
    if not final_png_path.exists():
        raise HTTPException(404, detail="final.png not found in folder")

    async def file_iterator():
        with open(final_png_path, "rb") as f:
            yield f.read()

    return StreamingResponse(
        file_iterator(),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=final.png"}
    )

@app.post("/translate/batch/json", response_model=list[TranslationResponse], tags=["api", "json", "batch"])
async def batch_json(req: Request, data: BatchTranslateRequest):
    """Batch translate images and return JSON format results"""
    results = await get_batch_ctx(req, data.config, data.images, data.batch_size)
    return [to_translation(ctx) for ctx in results]

@app.post("/translate/batch/images", response_description="Zip file containing translated images", tags=["api", "batch"])
async def batch_images(req: Request, data: BatchTranslateRequest):
    """Batch translate images and return zip archive containing translated images"""
    import zipfile
    import tempfile
    
    results = await get_batch_ctx(req, data.config, data.images, data.batch_size)
    
    # Create temporary ZIP file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        with zipfile.ZipFile(tmp_file, 'w') as zip_file:
            for i, ctx in enumerate(results):
                if ctx.result:
                    img_byte_arr = io.BytesIO()
                    ctx.result.save(img_byte_arr, format="PNG")
                    zip_file.writestr(f"translated_{i+1}.png", img_byte_arr.getvalue())
        
        # Return ZIP file
        with open(tmp_file.name, 'rb') as f:
            zip_data = f.read()
        
        # Clean up temporary file
        os.unlink(tmp_file.name)
        
        return StreamingResponse(
            io.BytesIO(zip_data),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=translated_images.zip"}
        )

@app.get("/", response_class=HTMLResponse,tags=["ui"])
async def index() -> HTMLResponse:
    script_directory = Path(__file__).parent
    html_file = script_directory / "index.html"
    html_content = html_file.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)

@app.get("/manual", response_class=HTMLResponse, tags=["ui"])
async def manual():
    script_directory = Path(__file__).parent
    html_file = script_directory / "manual.html"
    html_content = html_file.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)

def generate_nonce():
    return secrets.token_hex(16)

def start_translator_client_proc(host: str, port: int, nonce: str, params: Namespace):
    cmds = [
        sys.executable,
        '-m', 'manga_translator',
        'shared',
        '--host', host,
        '--port', str(port),
        '--nonce', nonce,
    ]
    if params.use_gpu:
        cmds.append('--use-gpu')
    if params.use_gpu_limited:
        cmds.append('--use-gpu-limited')
    if params.ignore_errors:
        cmds.append('--ignore-errors')
    if params.verbose:
        cmds.append('--verbose')
    if params.models_ttl:
        cmds.append('--models-ttl=%s' % params.models_ttl)
    if getattr(params, 'context_size', 0):
        cmds.append('--context-size=%s' % params.context_size)
    if getattr(params, 'pre_dict', None):
        cmds.extend(['--pre-dict', params.pre_dict])
    if getattr(params, 'post_dict', None):
        cmds.extend(['--post-dict', params.post_dict])       
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(base_path)
    proc = subprocess.Popen(cmds, cwd=parent)
    instance = ExecutorInstance(ip=host, port=port, reserve=getattr(params, 'lazy', False), slots=LOCAL_CHUNK_SLOTS)
    executor_instances.register(instance)
    _local_worker.update(proc=proc, instance=instance, cmds=cmds, cwd=parent)

    def handle_exit_signals(signal, frame):
        _local_worker['proc'].terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit_signals)
    signal.signal(signal.SIGTERM, handle_exit_signals)

    return proc


# The local GPU worker is a child of this process, and the auto-restart wrapper only watches this
# process. A worker that died (the system running out of memory kills it outright) used to leave
# the pool pointing at a dead port until someone restarted the whole server, while the interrupted
# gallery chunk burned its stall budget against the refused connection within seconds. Watch it:
# while it is down it leaves the pool, so a queued chunk waits (myqueue.NO_EXECUTOR_TIMEOUT_S)
# instead of failing, and a fresh worker goes back in once it is listening.
_local_worker: dict = {}
WORKER_RESTART_DELAY_S = 5
# Gallery chunks the local worker runs at once: a second one reads its pages while the first finishes
# its tail (see ExecutorInstance.slots). Must not exceed the worker's MT_WORKER_GALLERY_RUNS.
LOCAL_CHUNK_SLOTS = int(os.getenv('MT_LOCAL_CHUNK_SLOTS', '2'))
worker_logger = logging.getLogger('local-worker')
# Safety net for a worker whose committed memory has crept past this budget over a long uptime
# (fragmented C heap, allocator caches), measured after a chunk's own trim and including its process
# pool: it leaves the rotation, finishes the chunk it is on, and restarts — seconds of model loading
# instead of the machine running out of virtual memory mid-gallery. 0 turns it off.
WORKER_RECYCLE_GB = float(os.getenv('MT_WORKER_RECYCLE_GB', '28'))


def _check_worker_memory(telemetry: dict) -> None:
    mem = telemetry.get('mem') or {}
    if not mem.get('trimmed'):
        return   # judged right after a trim only: before one, commit still holds freeable caches
    commit = (mem.get('commit_end') or 0) + (mem.get('children_max') or 0)
    if WORKER_RECYCLE_GB <= 0 or _local_worker.get('recycle') or commit <= WORKER_RECYCLE_GB:
        return
    if mem.get('pid') not in {pid for pid, _ in _local_worker.get('tree', [])}:
        return   # a chunk from an aux node, not this machine's worker
    _local_worker['recycle'] = True
    executor_instances.unregister(_local_worker['instance'])
    worker_logger.warning(f'Local worker holds {commit:.1f} GB of committed memory (budget '
                          f'{WORKER_RECYCLE_GB:g} GB); restarting it once its current chunk finishes')


def _process_tree(pid: int) -> list:
    import psutil
    try:
        return [(p.pid, p.create_time()) for p in psutil.Process(pid).children(recursive=True)]
    except psutil.Error:
        return []


def _kill_survivors(tree: list) -> None:
    """A dead worker's process-pool children outlive it on Windows, each still holding a few GB
    of committed memory. Reclaim that before a replacement spawns its own set. The creation time
    keeps a recycled pid from matching a process that was never part of the worker."""
    import psutil
    for pid, created in tree:
        try:
            p = psutil.Process(pid)
            if p.create_time() == created:
                p.kill()
        except psutil.Error:
            pass


async def _worker_listening(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection(host, port)
    except OSError:
        return False
    writer.close()
    return True


async def _supervise_local_worker() -> None:
    while True:
        await asyncio.sleep(1)
        proc = _local_worker.get('proc')
        if proc is None:
            continue
        if proc.poll() is None:
            # Kept current while it runs, because once it is dead its children can no longer be
            # found: under a venv the spawned process is a launcher stub, and the process pool
            # hangs off the stub's child, the real worker.
            _local_worker['tree'] = await asyncio.to_thread(_process_tree, proc.pid)
            if _local_worker.get('recycle') and _local_worker['instance'].active == 0:
                # Its last chunk is done and nothing new can reach it: stop it; the exit path
                # below brings a fresh one up.
                await asyncio.to_thread(_kill_survivors, _local_worker['tree'])
            continue
        old = _local_worker['instance']
        executor_instances.unregister(old)
        if _local_worker.pop('recycle', False):
            worker_logger.info(f'Local worker stopped to release its memory; restarting it in '
                               f'{WORKER_RESTART_DELAY_S}s')
        else:
            worker_logger.error(f'Local worker exited (code {proc.returncode}); restarting it in '
                                f'{WORKER_RESTART_DELAY_S}s')
        await asyncio.to_thread(_kill_survivors, _local_worker.pop('tree', []))
        await asyncio.sleep(WORKER_RESTART_DELAY_S)
        proc = subprocess.Popen(_local_worker['cmds'], cwd=_local_worker['cwd'])
        _local_worker['proc'] = proc
        while proc.poll() is None and not await _worker_listening(old.ip, old.port):
            await asyncio.sleep(1)
        if proc.poll() is None:
            instance = ExecutorInstance(ip=old.ip, port=old.port, reserve=old.reserve, slots=old.slots)
            executor_instances.register(instance)
            _local_worker['instance'] = instance
            worker_logger.info(f'Local worker is back on {instance.label}')
            # Waiting tasks re-check on this rather than at their next 5-second tick.
            await task_queue.update_event()


@app.on_event("startup")
async def _start_worker_supervisor() -> None:
    if _local_worker:
        gallery_jobs.chunk_watchers.append(_check_worker_memory)
        asyncio.create_task(_supervise_local_worker())

def prepare(args):
    global nonce
    if args.nonce is None:
        nonce = os.getenv('MT_WEB_NONCE', generate_nonce())
    else:
        nonce = args.nonce
    if args.start_instance:
        return start_translator_client_proc(args.host, args.port + 1, nonce, args)
    folder_name= "upload-cache"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

@app.post("/simple_execute/translate_batch", tags=["internal-api"])
async def simple_execute_batch(req: Request, data: BatchTranslateRequest):
    """Internal batch translation execution endpoint"""
    # Implementation for batch translation logic
    # Currently returns empty results, actual implementation needs to call batch translator
    from manga_translator import MangaTranslator
    translator = MangaTranslator({'batch_size': data.batch_size})
    
    # Prepare image-config pairs
    images_with_configs = [(img, data.config) for img in data.images]
    
    # Execute batch translation
    results = await translator.translate_batch(images_with_configs, data.batch_size)
    
    return results

@app.post("/execute/translate_batch", tags=["internal-api"])
async def execute_batch_stream(req: Request, data: BatchTranslateRequest):
    """Internal batch translation streaming execution endpoint"""
    # Streaming batch translation implementation
    from manga_translator import MangaTranslator
    translator = MangaTranslator({'batch_size': data.batch_size})
    
    # Prepare image-config pairs
    images_with_configs = [(img, data.config) for img in data.images]
    
    # Execute batch translation (streaming version requires more complex implementation)
    results = await translator.translate_batch(images_with_configs, data.batch_size)
    
    return results

@app.get("/results/list", tags=["api"])
async def list_results():
    """List all result directories"""
    result_dir = RESULT_ROOT
    if not result_dir.exists():
        return {"directories": []}
    
    try:
        directories = []
        for item_path in result_dir.iterdir():
            if item_path.is_dir():
                # Check if final.png exists in this directory
                final_png_path = item_path / "final.png"
                if final_png_path.exists():
                    directories.append(item_path.name)
        return {"directories": directories}
    except Exception as e:
        raise HTTPException(500, detail=f"Error listing results: {str(e)}")

@app.delete("/results/clear", tags=["api"])
async def clear_results():
    """Delete all result directories"""
    result_dir = RESULT_ROOT
    if not result_dir.exists():
        return {"message": "No results directory found"}
    
    try:
        deleted_count = 0
        for item_path in result_dir.iterdir():
            if item_path.is_dir():
                # Check if final.png exists in this directory
                final_png_path = item_path / "final.png"
                if final_png_path.exists():
                    shutil.rmtree(item_path)
                    deleted_count += 1
        
        return {"message": f"Deleted {deleted_count} result directories"}
    except Exception as e:
        raise HTTPException(500, detail=f"Error clearing results: {str(e)}")

@app.delete("/results/{folder_name}", tags=["api"])
async def delete_result(folder_name: str):
    """Delete a specific result directory"""
    result_dir = RESULT_ROOT
    folder_path = result_dir / folder_name
    
    if not folder_path.exists():
        raise HTTPException(404, detail="Result directory not found")
    
    try:
        # Check if final.png exists in this directory
        final_png_path = folder_path / "final.png"
        if not final_png_path.exists():
            raise HTTPException(404, detail="Result file not found")
        
        shutil.rmtree(folder_path)
        return {"message": f"Deleted result directory: {folder_name}"}
    except Exception as e:
        raise HTTPException(500, detail=f"Error deleting result: {str(e)}")

#todo: restart if crash
#todo: cache results
#todo: cleanup cache

if __name__ == '__main__':
    import uvicorn
    from args import parse_arguments

    args = parse_arguments()

    if args.aux:
        # Auxiliary node: no public API, no job store, no scheduler — just a local worker and
        # the relay that feeds it from the main server. Exits non-zero on a fatal join refusal
        # (bad token / protocol / version) so a supervisor doesn't loop on it forever.
        from server import aux_agent
        if args.verbose:
            print('--verbose has no effect on an aux node: it shows nothing about the jobs it runs.')
        sys.exit(asyncio.run(aux_agent.run(args)))

    args.start_instance = True
    configure_stages(args)
    proc = prepare(args)
    print("Nonce: "+nonce)
    if args.lazy:
        # The worker still starts — it is the fallback for "every aux node went away", and
        # with --models-ttl it unloads from VRAM while idle, so the GPU is free in practice.
        print("Lazy mode: the local GPU is held in reserve and used only when no aux node is connected.")
        if not aux_pool.JOIN_TOKEN:
            print("  NOTE: MT_AUX_TOKEN is unset, so no aux node can join — everything will run "
                  "locally until you set it in .env and restart.")
    try:
        uvicorn.run(app, host=args.host, port=args.port,
                    ws_max_size=AUX_FRAME_BYTES)
    except Exception:
        if proc:
            _local_worker['proc'].terminate()
