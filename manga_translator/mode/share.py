import asyncio
import pickle
import io
import json
import secrets
from threading import Lock

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Path, Request, Response
from pydantic import BaseModel

from starlette.responses import StreamingResponse

from manga_translator import MangaTranslator
from manga_translator.utils.executors import run_cpu

SAFE_PICKLE_MODULES = frozenset({
    'builtins',
    'collections',
    'numpy',
    'numpy.core.multiarray',
    'numpy.dtype',
    'manga_translator',
    'manga_translator.utils',
    'manga_translator.utils.generic',
    'manga_translator.config'
})

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module in SAFE_PICKLE_MODULES or module.startswith('PIL.'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Deserialization of {module}.{name} is not allowed"
        )


def restricted_loads(data: bytes):
    return RestrictedUnpickler(io.BytesIO(data)).load()

class MethodCall(BaseModel):
    method_name: str
    attributes: bytes





class MangaShare:
    def __init__(self, params: dict = None):
        self.manga = MangaTranslator(params)
        self.host = params.get('host', '127.0.0.1')
        self.port = int(params.get('port', '5003'))
        nonce = params.get('nonce', None)
        if not nonce:
            nonce = secrets.token_hex(16)
        if nonce == "None":
            nonce = None
        self.nonce = nonce

        # each chunk has a structure like this status_code(int/1byte),len(int/4bytes),bytechunk
        # status codes are 0 for result, 1 for progress report, 2 for error
        self.progress_queue = asyncio.Queue()
        self.lock = Lock()

        async def hook(state: str, finished: bool):
            state_data = state.encode("utf-8")
            progress_data = b'\x01' + len(state_data).to_bytes(4, 'big') + state_data
            await self.progress_queue.put(progress_data)
            await asyncio.sleep(0)

        self.manga.add_progress_hook(hook)

        # status 5 = one finished page during a streaming batch (gallery) translation:
        # data = tokenLen(1 byte) + job_token(utf-8) + page-index (4 bytes BE) + PNG bytes.
        # The token lets the client reject any page frame that isn't its own job's — a
        # second, independent guard against cross-gallery mix-up on top of the per-job
        # queue isolation. Lets the client render pages as they complete.
        async def page_result_hook(index: int, image):
            if image is None:
                return
            # Encode the finished page as WebP — a translated manga page is a fraction of the
            # PNG size at quality the eye can't tell from lossless (the source pages are WebP
            # too). Fall back to PNG if this Pillow build lacks WebP. The client sniffs the
            # bytes, so either format is stored with the right type. Encoding a full page is
            # hundreds of ms, so it runs on the CPU pool (method=4: ~3× faster than 6, ~same
            # size at q95), never on the orchestrating loop.
            def _encode():
                img_byte_arr = io.BytesIO()
                try:
                    image.save(img_byte_arr, format="WEBP", quality=95, method=4)
                except Exception:
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="PNG")
                return img_byte_arr.getvalue()
            png = await run_cpu(_encode)
            token = (getattr(self.manga, '_gallery_job_token', '') or '').encode('utf-8')[:255]
            data = bytes([len(token)]) + token + index.to_bytes(4, 'big') + png
            frame = b'\x05' + len(data).to_bytes(4, 'big') + data
            await self.progress_queue.put(frame)
            await asyncio.sleep(0)

        self.manga.add_page_result_hook(page_result_hook)

        # status 6 = one page's study layers. Same envelope as status 5: data = tokenLen(1) +
        # job_token + page-index (4 BE) + JSON bytes, where the JSON is {bg, bubbles}: bg is a
        # base64 PNG data URL (the inpainted page) and bubbles is a list of {box, region, tr,
        # src, text} — box/region are page-fraction rects (the OCR border and the bg-clip area)
        # and text is a base64 full-page transparent PNG of just that bubble's glyphs. The token
        # is the same cross-gallery mix-up guard.
        async def page_bubbles_hook(index: int, bubbles: dict):
            # The payload embeds base64 image layers (can be several MB) — serialize off-loop.
            payload = await run_cpu(lambda: json.dumps(bubbles, separators=(',', ':')).encode('utf-8'))
            token = (getattr(self.manga, '_gallery_job_token', '') or '').encode('utf-8')[:255]
            data = bytes([len(token)]) + token + index.to_bytes(4, 'big') + payload
            frame = b'\x06' + len(data).to_bytes(4, 'big') + data
            await self.progress_queue.put(frame)
            await asyncio.sleep(0)

        self.manga.add_page_bubbles_hook(page_bubbles_hook)

    async def progress_stream(self, q):
        """
        Streams progress (1) and per-page result (5) frames, terminating only on the
        final result (0) or an error (2).

        Reads from the queue `q` handed in by /execute — one fresh queue per job — so a
        previous job's undrained frames (e.g. its consumer died mid-gallery) live on an
        orphaned queue object and can NEVER be delivered into the next job's stream.
        """
        while True:
            progress = await q.get()
            yield progress
            if progress[0] == 0 or progress[0] == 2:
                break

    async def run_method(self, method, **attributes):
        try:
            if asyncio.iscoroutinefunction(method):
                result = await method(**attributes)
            else:
                result = method(**attributes)

            # 检查是否使用占位符，如果是则创建最小化的结果对象
            if hasattr(result, 'use_placeholder') and result.use_placeholder:
                # 创建一个最小的Context对象，只包含占位符图片，避免传输大量数据
                from manga_translator import Context
                from PIL import Image
                minimal_result = Context()
                minimal_result.result = Image.new('RGB', (1, 1), color='white')
                minimal_result.use_placeholder = True
                result_bytes = pickle.dumps(minimal_result)
            else:
                result_bytes = pickle.dumps(result)

            encoded_result = b'\x00' + len(result_bytes).to_bytes(4, 'big') + result_bytes
            await self.progress_queue.put(encoded_result)
        except Exception as e:
            err_bytes = str(e).encode("utf-8")
            encoded_result = b'\x02' + len(err_bytes).to_bytes(4, 'big') + err_bytes
            await self.progress_queue.put(encoded_result)
        finally:
            self.lock.release()


    def check_nonce(self, request: Request):
        if self.nonce:
            nonce = request.headers.get('X-Nonce')
            if nonce != self.nonce:
                raise HTTPException(401, detail="Nonce does not match")

    def check_lock(self):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="some Method is already being executed.")

    def get_fn(self, method_name: str):
        if method_name.startswith("__"):
            raise HTTPException(status_code=403, detail="These functions are not allowed to be executed remotely")
        method = getattr(self.manga, method_name, None)
        if not method:
            raise HTTPException(status_code=404, detail="Method not found")
        return method

    def build_app(self):
        """Build the worker FastAPI app. Split out from listen() so tests can drive the
        routes (e.g. token-scoped /cancel_gallery) with a TestClient — no uvicorn needed."""
        app = FastAPI()

        @app.get("/is_locked")
        async def is_locked():
            if self.lock.locked():
                return {"locked": True}
            return {"locked": False}

        @app.post("/cancel_gallery")
        async def cancel_gallery(request: Request, job_token: str = Form("")):
            """Abort the in-flight gallery pipeline (explicit cancel or client disconnect).
            Deliberately skips check_lock — the lock is held by the very job we abort.

            Token-scoped: only cancels when job_token matches the running job's token (or
            when no token is given, the legacy "cancel whatever is running"). This stops a
            late cancel from killing a *different* gallery that started in the meantime."""
            self.check_nonce(request)
            running = getattr(self.manga, '_gallery_job_token', '') or ''
            if not job_token or job_token == running:
                self.manga._gallery_cancel = True
                return {"cancelling": self.lock.locked()}
            return {"cancelling": False}

        @app.post("/simple_execute/{method_name}")
        async def execute_method(request: Request, method_name: str = Path(...)):
            self.check_nonce(request)
            self.check_lock()
            method = self.get_fn(method_name)
            attr = restricted_loads(await request.body())
            try:
                if asyncio.iscoroutinefunction(method):
                    result = await method(**attr)
                else:
                    result = method(**attr)
                self.lock.release()
                result_bytes = pickle.dumps(result)
                return Response(content=result_bytes, media_type="application/octet-stream")
            except Exception as e:
                self.lock.release()
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/execute/{method_name}")
        async def execute_method(request: Request, method_name: str = Path(...)):
            self.check_nonce(request)
            self.check_lock()
            method = self.get_fn(method_name)
            attr = restricted_loads(await request.body())

            # 根据端点类型决定是否使用占位符优化
            config = attr.get('config')
            self.manga._is_streaming_mode = getattr(config, '_web_frontend_optimized', False) if config else False

            # Fresh queue per job (check_lock serialises jobs, so the hooks — which push to
            # self.progress_queue — always target the active job's queue). The stream reads
            # this exact `q`, so a dead/aborted job's leftover frames can't bleed into the
            # next job's response. This is the structural guard against cross-gallery mix-up.
            q = asyncio.Queue()
            self.progress_queue = q

            # streaming response
            streaming_response = StreamingResponse(self.progress_stream(q), media_type="application/octet-stream")
            asyncio.create_task(self.run_method(method, **attr))
            return streaming_response

        return app

    async def listen(self, translation_params: dict = None):
        config = uvicorn.Config(self.build_app(), host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()
