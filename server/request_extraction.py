import asyncio
import builtins
import io
import re
from base64 import b64decode
from typing import Union

import requests
from PIL import Image
from fastapi import Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from manga_translator import Config
from server.myqueue import task_queue, wait_in_queue, QueueElement, BatchQueueElement
from server.streaming import notify, stream
from server import gallery_jobs

class TranslateRequest(BaseModel):
    """This request can be a multipart or a json request"""
    image: bytes|str
    """can be a url, base64 encoded image or a multipart image"""
    config: Config = Config()
    """in case it is a multipart this needs to be a string(json.stringify)"""

class BatchTranslateRequest(BaseModel):
    """Batch translation request"""
    images: list[bytes|str]
    """List of images, can be URLs, base64 encoded strings, or binary data"""
    config: Config = Config()
    """Translation configuration"""
    batch_size: int = 4
    """Batch size, default is 4"""

async def to_pil_image(image: Union[str, bytes]) -> Image.Image:
    try:
        if isinstance(image, builtins.bytes):
            image = Image.open(io.BytesIO(image))
            return image
        else:
            if re.match(r'^data:image/.+;base64,', image):
                value = image.split(',', 1)[1]
                image_data = b64decode(value)
                image = Image.open(io.BytesIO(image_data))
                return image
            else:
                response = requests.get(image)
                image = Image.open(io.BytesIO(response.content))
                return image
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


async def get_ctx(req: Request, config: Config, image: str|bytes):
    image = await to_pil_image(image)

    task = QueueElement(req, image, config, 0)
    task_queue.add_task(task)

    return await wait_in_queue(task, None)

async def while_streaming(req: Request, transform, config: Config, image: bytes | str):
    image = await to_pil_image(image)

    task = QueueElement(req, image, config, 0)
    task_queue.add_task(task)

    messages = asyncio.Queue()

    def notify_internal(code: int, data: bytes) -> None:
        notify(code, data, transform, messages)
    streaming_response = StreamingResponse(stream(messages), media_type="application/octet-stream")
    asyncio.create_task(wait_in_queue(task, notify_internal))
    return streaming_response

async def start_gallery_job(req: Request, transform, config: Config, images: list[bytes | str], batch_size: int = 0, job_token: str = ""):
    """Polling model: create the server-owned job and hand it to the chunk scheduler, then
    return IMMEDIATELY. The scheduler dispatches the job to the worker one chunk of pages at
    a time (rotating chunks between concurrent jobs — see gallery_jobs), buffering every frame
    into the job; the client collects them with short /translate/gallery/poll requests instead
    of holding one long stream open (which a service worker can't, because of the ~5-min
    per-event lifetime cap). Idempotent on job_token."""
    existing = gallery_jobs.get(job_token)
    if existing is not None:
        return {"token": job_token, "started": True, "existing": True}

    job = gallery_jobs.create(job_token)
    job.total = len(images)               # authoritative page count for the poll progress bar
    gallery_jobs.submit(job, req, images, config, batch_size, transform)
    return {"token": job_token, "started": True}

async def get_batch_ctx(req: Request, config: Config, images: list[str|bytes], batch_size: int = 4):
    """Process batch translation request"""
    # Convert images to PIL Image objects
    pil_images = []
    for img in images:
        pil_img = await to_pil_image(img)
        pil_images.append(pil_img)
    
    # Create batch task
    batch_task = BatchQueueElement(req, pil_images, config, batch_size)
    task_queue.add_task(batch_task)
    
    return await wait_in_queue(batch_task, None)