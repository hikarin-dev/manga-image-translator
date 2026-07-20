import io
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
from argparse import Namespace
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fastapi import FastAPI, Request, HTTPException, Header, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from manga_translator import Config
from server import aux_pool
from server import edge
from server import stats
from server.instance import ExecutorInstance, executor_instances
from server.myqueue import task_queue, running_galleries, GalleryQueueElement
from server import gallery_jobs
from server.request_extraction import get_ctx, while_streaming, start_gallery_job, TranslateRequest, BatchTranslateRequest, get_batch_ctx
from server.to_json import to_translation, TranslationResponse

app = FastAPI()
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
async def dashboard() -> HTMLResponse:
    """Operator dashboard: pool state, queue, today's totals, recent jobs.

    Always available on loopback. Through the tunnel it is address-gated rather than
    token-gated (a browser navigation cannot carry X-Access-Token): reachable from
    MT_DASHBOARD_IPS and from whichever aux nodes are connected — see server.edge."""
    return HTMLResponse(content=(BASE_DIR / "dashboard.html").read_text(encoding="utf-8"))

@app.get("/dashboard/data", tags=["ui"])
async def dashboard_data() -> dict:
    """Everything the dashboard polls, in one request so its numbers are from one instant."""
    gpu = await asyncio.to_thread(stats.gpu_snapshot)
    return {**stats.snapshot(gpu), "executors": aux_pool.nodes()}

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

@app.post("/translate/gallery/start", tags=["api", "form", "batch"], response_description="Create a server-owned gallery job and return immediately with its token; collect results via /translate/gallery/poll. A big gallery may arrive as several requests sharing one token (part k of n) — the job starts when the last part lands.")
async def start_gallery(req: Request, image: list[UploadFile] = File(...), config: str = Form("{}"), batch_size: int = Form(0), job_token: str = Form(""), part: int = Form(0), parts: int = Form(1)) -> dict:
    images = [await f.read() for f in image]
    external = bool(getattr(req.state, "external", False))
    client_ip = str(getattr(req.state, "client_ip", "") or "")
    if external:
        err = edge.validate_pages(images)
        if err:
            raise HTTPException(413, detail=err)
    conf = Config.parse_raw(config)
    if parts > 1:
        if not (job_token and 1 < parts <= 200 and 0 <= part < parts):
            raise HTTPException(400, detail="bad part/parts")
        status, assembled = gallery_jobs.add_upload_part(
            job_token, part, parts, images, client_ip,
            max_pages=edge.MAX_PAGES_PER_JOB if external else 0)
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
    return await start_gallery_job(req, transform_gallery_summary, conf, images, batch_size, job_token)

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

@app.post("/queue-size", response_model=int, tags=["api", "json"])
async def queue_size() -> int:
    return len(task_queue.queue)

@app.get("/stats", tags=["api"])
async def service_stats() -> dict:
    """Operator metrics: queue depth, today's jobs/pages, GPU state, recent job summaries.
    Externally reachable (token-gated by the edge middleware); per-job history persists in
    logs/jobs.jsonl."""
    gpu = await asyncio.to_thread(stats.gpu_snapshot)
    return stats.snapshot(gpu)

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
    executor_instances.register(ExecutorInstance(ip=host, port=port,
                                                 reserve=getattr(params, 'lazy', False)))

    def handle_exit_signals(signal, frame):
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit_signals)
    signal.signal(signal.SIGTERM, handle_exit_signals)

    return proc

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
        sys.exit(asyncio.run(aux_agent.run(args)))

    args.start_instance = True
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
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception:
        if proc:
            proc.terminate()
