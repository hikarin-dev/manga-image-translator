"""Measure fragmented snapshot-frame assembly with real archived assets."""
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.snapshots import export_archive
from server.sent_data_internal import process_stream
import hashlib


async def main():
    row = json.loads(Path('dev/performance-20260922/index.json').read_text())[4]
    manifest = Path(row['manifest']).read_bytes()
    m = json.loads(manifest)
    payload = export_archive(m['run_id'], hashlib.sha256(manifest).hexdigest())
    frame = b'\x08' + len(payload).to_bytes(4, 'big') + payload
    result = {'frame_bytes': len(frame), 'chunk_bytes': 65536, 'repeats': []}
    for i in range(6):
        implementation = 'before' if i % 2 == 0 else 'after'
        chunks = [frame[n:n+65536] for n in range(0, len(frame), 65536)]
        seen = []
        start = time.perf_counter()
        if implementation == 'before':
            buffer = b''
            for chunk in chunks:
                buffer += chunk
                if len(buffer) >= 5:
                    size = int.from_bytes(buffer[1:5], 'big')
                    if len(buffer) >= size+5:
                        seen.append((buffer[0], buffer[5:5+size]))
                        buffer = buffer[5+size:]
                await asyncio.sleep(0)
        else:
            reader = asyncio.StreamReader()
            task = asyncio.create_task(process_stream(SimpleNamespace(content=reader), lambda *data: seen.append(data)))
            for chunk in chunks:
                reader.feed_data(chunk)
                await asyncio.sleep(0)
            reader.feed_eof()
            await task
        elapsed = time.perf_counter()-start
        assert len(seen) == 1 and seen[0] == (8, payload)
        result['repeats'].append({'implementation': implementation, 'seconds': elapsed})
    Path('dev/performance-20260922/transport.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    asyncio.run(main())
