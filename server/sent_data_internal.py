import json
import pickle
from typing import Mapping, Optional, Callable

import aiohttp
from PIL.Image import Image
from fastapi import HTTPException

from manga_translator import Config

NotifyType = Optional[Callable[[int, Optional[bytes]], None]]

# A streaming translation (a whole gallery, or one big page) legitimately runs far longer than
# aiohttp's DEFAULT 5-minute total timeout — a slow LLM batch alone can exceed it. That default
# silently killed long gallery jobs mid-way (asyncio.TimeoutError → "Translation failed:") while
# the worker, never told to stop, kept churning the GPU into the dead connection. We cap only
# connection setup and idle gaps (sock_read), never total runtime, so a job runs as long as it
# keeps making progress; a truly hung worker still trips sock_read instead of hanging forever.
_STREAM_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=900)

async def fetch_data_stream(url, image: Image, config: Config, sender: NotifyType, headers: Mapping[str, str] = {}):
    attributes = {"image": image, "config": config}
    data = pickle.dumps(attributes)

    async with aiohttp.ClientSession(timeout=_STREAM_TIMEOUT) as session:
        async with session.post(url, data=data, headers=headers) as response:
            if response.status == 200:
                await process_stream(response, sender)
            else:
                raise HTTPException(response.status, detail=await response.text())

async def fetch_gallery_stream(url, images: list, config: Config, sender: NotifyType, batch_size: int = 0, job_token: str = "", headers: Mapping[str, str] = {}):
    """Stream a whole-gallery batch translation. Pickles all page images (kept as
    compressed bytes; the worker decodes them one at a time) + config to the worker's
    translate_gallery_stream; the worker streams progress (1), per-page result (5)
    and a final summary (0) frames, handled the same way as a single image."""
    attributes = {"images": images, "config": config, "batch_size": batch_size, "job_token": job_token}
    data = pickle.dumps(attributes)

    async with aiohttp.ClientSession(timeout=_STREAM_TIMEOUT) as session:
        async with session.post(url, data=data, headers=headers) as response:
            if response.status == 200:
                await process_stream(response, sender)
            else:
                raise HTTPException(response.status, detail=await response.text())

async def post_cancel(url, job_token: str = "", headers: Mapping[str, str] = {}):
    """Best-effort fire-and-forget POST (gallery cancellation); never raises.
    Sends the job_token as a form field so the worker can verify it's cancelling
    the right job."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data={"job_token": job_token}, headers=headers):
                pass
    except Exception:
        pass

async def fetch_data(url, image: Image, config: Config, headers: Mapping[str, str] = {}):
    attributes = {"image": image, "config": config}
    data = pickle.dumps(attributes)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers=headers) as response:
            if response.status == 200:
                try:
                    return json.loads(await response.text())
                except json.JSONDecodeError:
                    raise HTTPException(502, detail='Invalid JSON response from upstream')
            else:
                raise HTTPException(response.status, detail=await response.text())

async def process_stream(response, sender: NotifyType):
    buffer = b''

    async for chunk in response.content.iter_any():
        if chunk:
            buffer += chunk
            buffer = handle_buffer(buffer, sender)



def handle_buffer(buffer, sender: NotifyType):
    while len(buffer) >= 5:
        status, expected_size = extract_header(buffer)

        if len(buffer) >= 5 + expected_size:
            data = buffer[5:5 + expected_size]
            sender(status, data)
            buffer = buffer[5 + expected_size:]
        else:
            break
    return buffer


def extract_header(buffer):
    """Extract the status and expected size from the buffer."""
    status = int.from_bytes(buffer[0:1], byteorder='big')
    expected_size = int.from_bytes(buffer[1:5], byteorder='big')
    return status, expected_size

