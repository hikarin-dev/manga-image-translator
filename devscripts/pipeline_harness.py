"""Measurement harness for the pipeline refactor. Drives a running server through the normal gallery
API with pages from dev/corpus-benchmark (see its index.json) and writes one JSON report.

  long   one gallery job of --pages pages (the corpus cycled in order); per-page arrival times and
         a 2 s sample of the worker's memory, system commit, CPU and GPU
  fair   a long job, then after --delay seconds a --small-page job; both jobs' first-page and
         completion times, for the scheduling SLA
  equiv  one job; every page, Study and pipeline-data frame saved (next to --out) and hashed, for
         comparing two code versions (use a deterministic translator such as sugoi). Compare with
         `pipeline_harness.py diff A.json B.json`: identical, within the known noise (detector soft
         map ±8 levels, line scores ±1e-5, composited Study page ≤ 0.1% of pixels), or different

The baseline settings are the app's (detector default, Hayai, DeepSeek at cap 10, Lama Large,
shiori, Study text_only, snapshots off); every one of them can be overridden.
"""
import argparse
import asyncio
import ctypes
import hashlib
import json
import subprocess
import sys
import threading
import time
import uuid
from ctypes import wintypes
from pathlib import Path

import aiohttp
import psutil

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / 'dev' / 'corpus-benchmark'

BASELINE = {
    'detector': {'detector': 'default', 'detection_size': 2560, 'text_threshold': 0.5, 'box_threshold': 0.75,
                 'unclip_ratio': 2.3},
    'ocr': {'ocr': 'hayai'},
    'render': {'estimate_font_color': False, 'estimate_outline_color': False, 'renderer': 'shiori',
               'direction': 'auto', 'alignment': 'auto', 'font_size_offset': 0, 'uppercase': False,
               'no_hyphenation': False},
    'translator': {'translator': 'deepseek', 'target_lang': 'ENG', 'enable_post_translation_check': False},
    'inpainter': {'inpainter': 'lama_large', 'inpainting_size': 2048, 'inpainting_precision': 'bf16'},
    'mask_dilation_offset': 40, 'kernel_size': 7, 'study_mode_generation': 'text_only',
}


# ── sampling ──────────────────────────────────────────────────────────────────────────────────
class _PerfInfo(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
        'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable', 'SystemCache',
        'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
        (n, wintypes.DWORD) for n in ('HandleCount', 'ProcessCount', 'ThreadCount')]


def _system_commit() -> tuple[float, float]:
    if sys.platform != 'win32':
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 2**30, vm.total / 2**30
    pi = _PerfInfo()
    pi.cb = ctypes.sizeof(pi)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    return pi.CommitTotal * pi.PageSize / 2**30, pi.CommitLimit * pi.PageSize / 2**30


def _worker(port: str):
    for p in psutil.process_iter(['cmdline']):
        c = p.info['cmdline'] or []
        if 'shared' in c and port in c and not c[0].endswith(('Scripts\\python.exe', 'Scripts/python.exe')):
            return p
    return None


class Sampler(threading.Thread):
    def __init__(self, port: str, interval: float = 2.0):
        super().__init__(daemon=True)
        self.port, self.interval, self.rows, self._stop = port, interval, [], threading.Event()

    def run(self):
        psutil.cpu_percent()
        p = None
        t0 = time.monotonic()
        while not self._stop.wait(self.interval):
            row = {'t': round(time.monotonic() - t0, 1)}
            try:
                if p is None or not p.is_running():
                    p = _worker(self.port)
                if p is not None:
                    info = p.memory_info()
                    kids = sum(getattr(k.memory_info(), 'private', 0) for k in p.children(recursive=True))
                    row.update(pid=p.pid, commit=round(getattr(info, 'private', info.vms) / 2**30, 2),
                               rss=round(info.rss / 2**30, 2), threads=p.num_threads(),
                               children=round(kids / 2**30, 2))
                row['sys_commit'], row['sys_limit'] = (round(x, 1) for x in _system_commit())
                row['cpu'] = psutil.cpu_percent()
                g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                u, m = g.stdout.strip().splitlines()[0].split(', ')
                row['gpu'], row['vram_mb'] = float(u), float(m)
            except Exception as e:  # a sample is best effort; never kill the run over it
                row['error'] = str(e)
            self.rows.append(row)

    def stop(self):
        self._stop.set()


# ── jobs ──────────────────────────────────────────────────────────────────────────────────────
def corpus_pages(n: int, galleries: list[str] | None, spread: bool = False, samples: bool = False) -> list[Path]:
    """`n` pages in corpus order (cycling), or with `spread` evenly spaced across it. `samples`
    keeps only the pages the Settings benchmark measured (20 per benchmark gallery). Galleries
    marked `extra` in the index are used only when named, so adding one leaves the default page
    selection, and every reference run made with it, unchanged."""
    index = json.loads((CORPUS / 'index.json').read_text())
    files = [CORPUS / r['file'] for r in index
             if (r['gallery'] in galleries if galleries else not r.get('extra'))
             and (not samples or r['benchmark_sample'])]
    if not files:
        sys.exit('no corpus pages; see dev/corpus-benchmark')
    if spread and n <= len(files):
        return [files[round(i * (len(files) - 1) / max(1, n - 1))] for i in range(n)]
    return [files[i % len(files)] for i in range(n)]


def build_config(args) -> dict:
    cfg = json.loads(json.dumps(BASELINE))
    for override in args.set or []:
        path, value = override.split('=', 1)
        target = cfg
        *parents, leaf = path.split('.')
        for part in parents:
            target = target.setdefault(part, {})
        try:
            target[leaf] = json.loads(value)
        except json.JSONDecodeError:
            target[leaf] = value
    return cfg


async def run_job(client, args, cfg, pages: list[Path], label: str, hashes: bool = False) -> dict:
    token = uuid.uuid4().hex
    form = aiohttp.FormData()
    form.add_field('config', json.dumps(cfg))
    form.add_field('job_token', token)
    form.add_field('batch_size', str(args.cap))
    form.add_field('capture', 'true' if args.capture else 'false')
    form.add_field('source_url', f'harness:{label}')
    for p in pages:
        form.add_field('image', p.read_bytes(), filename=p.name, content_type='application/octet-stream')
    submitted = time.monotonic()
    async with client.post(args.server + '/translate/gallery/start', data=form) as r:
        r.raise_for_status()
    started = time.monotonic()
    cursor, arrivals, terminal = 0, {}, None
    frames: dict[int, dict[str, list[str]]] = {}
    while terminal is None:
        await asyncio.sleep(args.poll)
        async with client.post(args.server + '/translate/gallery/poll',
                               data={'job_token': token, 'since': str(cursor)}) as r:
            raw = await r.read()
        o = 0
        while o < len(raw):
            st, size = raw[o], int.from_bytes(raw[o + 1:o + 5], 'big')
            data = raw[o + 5:o + 5 + size]
            o += 5 + size
            if st == 7:
                meta = json.loads(data)
                cursor = meta['cursor']
                if meta.get('status') in ('notfound', 'cancelled'):
                    terminal = {'error': meta['status']}
            elif st in (5, 6, 9):
                b = 1 + data[0]
                idx = int.from_bytes(data[b:b + 4], 'big')
                if st == 5:
                    arrivals.setdefault(idx, round(time.monotonic() - started, 2))
                if hashes:
                    frames.setdefault(idx, {}).setdefault(str(st), []).append(hashlib.sha256(data[b + 4:]).hexdigest())
                if hashes:
                    n = len(frames[idx][str(st)])
                    dump = _frames_dir(args.out)
                    dump.mkdir(parents=True, exist_ok=True)
                    (dump / f'{idx:04d}-{st}-{n}.bin').write_bytes(data[b + 4:])
            elif st == 0:
                terminal = json.loads(data)
            elif st == 2:
                terminal = {'error': data.decode('utf-8', 'replace')}
    wall = time.monotonic() - started
    async with client.get(args.server + '/dashboard/data') as r:
        dash = await r.json()
    record = next((j for j in dash.get('recent_jobs', []) if j.get('token') == token[:8]), {})
    record = {k: v for k, v in record.items() if k not in ('ip', 'key', 'token')}
    out = {'label': label, 'pages': len(pages), 'delivered': len(arrivals), 'upload_s': round(started - submitted, 2),
           'wall_s': round(wall, 1), 's_per_page': round(wall / len(pages), 3),
           'first_page_s': min(arrivals.values()) if arrivals else None,
           'arrivals': [arrivals.get(i) for i in range(len(pages))], 'terminal': terminal, 'server': record}
    if hashes:
        out['frames'] = {str(k): v for k, v in sorted(frames.items())}
    return out


async def main(args):
    if args.mode == 'diff':
        return diff(args.a, args.b)
    cfg = build_config(args)
    pages = corpus_pages(args.pages + args.offset, args.gallery, args.spread, args.samples)[args.offset:]
    report = {'mode': args.mode, 'server': args.server, 'config': cfg, 'cap': args.cap, 'capture': args.capture,
              'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'pages': [p.relative_to(CORPUS).as_posix() for p in pages]}
    sampler = Sampler(args.worker_port)
    sampler.start()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as client:
            async with client.get(args.server + '/dashboard/data') as r:
                q = (await r.json())['queue']
            if q['live_jobs'] and not args.allow_busy:
                sys.exit(f'server busy ({q}); rerun when idle or pass --allow-busy')
            if args.warm:
                await run_job(client, args, cfg, corpus_pages(args.warm, args.gallery)[::-1], 'warm')
            if args.mode in ('long', 'equiv'):
                report['jobs'] = [await run_job(client, args, cfg, pages, args.mode, hashes=args.mode == 'equiv')]
            elif args.mode == 'fair':
                long_task = asyncio.create_task(run_job(client, args, cfg, pages, 'fair-long'))
                await asyncio.sleep(args.delay)
                small = corpus_pages(args.small, args.gallery)[-args.small:]
                small_job = await run_job(client, args, cfg, small, 'fair-small')
                small_job['submitted_after_s'] = args.delay
                report['jobs'] = [await long_task, small_job]
    finally:
        sampler.stop()
        report['samples'] = sampler.rows
    Path(args.out).write_text(json.dumps(report, indent=1))
    for j in report['jobs']:
        print(f"{j['label']}: {j['delivered']}/{j['pages']} pages, {j['s_per_page']} s/page, "
              f"first page {j['first_page_s']} s, wall {j['wall_s']} s, terminal {str(j['terminal'])[:80]}")
    mem = [r for r in sampler.rows if 'commit' in r]
    if mem:
        print(f"worker commit max {max(r['commit'] for r in mem)} GB, children max {max(r['children'] for r in mem)} GB, "
              f"system commit max {max(r['sys_commit'] for r in mem)} GB, GPU avg "
              f"{sum(r.get('gpu', 0) for r in mem) / len(mem):.0f}%")


STUDY_NOISE = 0.001   # composited Study pages may differ in this share of pixels (see plans: shiori renderer)


def _frames_dir(report_path) -> Path:
    return Path(report_path).with_suffix('')


def _study_page(payload: bytes):
    """Study metadata without its images, and the page as a reader shows it (bg + text layers)."""
    import base64
    import io
    import numpy as np
    from PIL import Image
    study = json.loads(payload)
    img = lambda u: Image.open(io.BytesIO(base64.b64decode(u.split(',', 1)[1])))
    meta = json.loads(json.dumps(study))
    meta.pop('bg', None)
    for bubble in meta.get('bubbles', []):
        bubble.pop('text', None)
    if not study.get('bg'):
        return meta, None
    page = img(study['bg']).convert('RGBA')
    for bubble in study.get('bubbles', []):
        if bubble.get('text'):
            page.alpha_composite(img(bubble['text']).convert('RGBA'))
    return meta, np.asarray(page.convert('RGB')).astype(int)


def _regions_close(a: dict, b: dict, tol: float = 0.005) -> bool:
    """Study metadata equal except `region` extents within `tol` of the page. text_and_image derives
    a bubble's region from its rendered glyph pixels, so the shiori renderer's run-to-run pixel
    noise can move it by a few px. manga2eng is deterministic: check strictly with it."""
    import copy
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    ba, bb = a.get('bubbles', []), b.get('bubbles', [])
    if len(ba) != len(bb):
        return False
    for x, y in zip(ba, bb):
        rx, ry = x.pop('region', {}), y.pop('region', {})
        if set(rx) != set(ry) or any(abs(rx[k] - ry[k]) > tol for k in rx):
            return False
    return a == b


def _pipeline_equal(a: bytes, b: bytes) -> tuple[bool, str]:
    """Pipeline data equal up to the detector's GPU float noise: soft map within 8 levels,
    line scores within 1e-5. Every other field, and the text mask, must match exactly."""
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from manga_translator import page_data as pd
    (ra, ba), (rb, bb) = pd.unpack(a), pd.unpack(b)
    notes = []
    for line_a, line_b in zip(ra.get('lines', []), rb.get('lines', [])):
        sa, sb = line_a.pop('score', None), line_b.pop('score', None)
        if (sa is None) != (sb is None) or (sa is not None and abs(float(sa) - float(sb)) > 1e-5):
            return False, 'line score'
    if ra != rb:
        return False, 'record'
    ra_raw, rb_raw = ba.pop('raw', None), bb.pop('raw', None)
    if ba != bb:
        return False, 'masks ' + ','.join(k for k in set(ba) | set(bb) if ba.get(k) != bb.get(k))
    if (ra_raw is None) != (rb_raw is None):
        return False, 'raw presence'
    if ra_raw is not None and ra_raw != rb_raw:
        d = np.abs(pd.decode_mask(ra_raw).astype(int) - pd.decode_mask(rb_raw).astype(int))
        if d.max() > 8:
            return False, f'raw map differs by {int(d.max())}'
        notes.append(f'raw map noise ({int((d > 0).sum())} px)')
    return True, '; '.join(notes)


def diff(a_path, b_path) -> bool:
    """Compare two equiv reports frame by frame: identical, within known noise, or different."""
    import numpy as np
    a, b = (json.loads(Path(p).read_text())['jobs'][0]['frames'] for p in (a_path, b_path))
    da, db = _frames_dir(a_path), _frames_dir(b_path)
    counts = {'identical': 0, 'noise': 0, 'different': 0}
    for k in sorted(set(a) | set(b), key=int):
        verdict, notes = 'identical', []
        for kind in sorted(set(a.get(k, {})) | set(b.get(k, {}))):
            ha, hb = a.get(k, {}).get(kind, []), b.get(k, {}).get(kind, [])
            if ha == hb:
                continue
            if len(ha) != len(hb):
                verdict = 'different'; notes.append(f'frame {kind} count {len(ha)} vs {len(hb)}'); continue
            for n, (x, y) in enumerate(zip(ha, hb), 1):
                if x == y:
                    continue
                pa = (da / f'{int(k):04d}-{kind}-{n}.bin').read_bytes()
                pb = (db / f'{int(k):04d}-{kind}-{n}.bin').read_bytes()
                if kind == '9':
                    ok, why = _pipeline_equal(pa, pb)
                elif kind == '6':
                    (ma, ia), (mb, ib) = _study_page(pa), _study_page(pb)
                    if ma != mb and not _regions_close(ma, mb):
                        ok, why = False, 'study metadata'
                    elif ia is None or ia.shape != ib.shape:
                        ok, why = False, 'study image'
                    else:
                        px = np.abs(ia - ib).max(axis=-1) > 0
                        ok = px.mean() <= STUDY_NOISE
                        why = f'study page {int(px.sum())} px ({px.mean():.4%})'
                else:
                    ok, why = False, 'page image'
                notes.append(why)
                if not ok:
                    verdict = 'different'
                elif verdict == 'identical':
                    verdict = 'noise'
        counts[verdict] += 1
        if verdict != 'identical':
            print(f'  page {k}: {verdict} - ' + '; '.join(n for n in notes if n))
    print(counts)
    return counts['different'] == 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['long', 'fair', 'equiv', 'diff'])
    ap.add_argument('a', nargs='?'); ap.add_argument('b', nargs='?')
    ap.add_argument('--server', default='http://127.0.0.1:5003')
    ap.add_argument('--worker-port', default='5004')
    ap.add_argument('--out', default='harness-report.json')
    ap.add_argument('--pages', type=int, default=400)
    ap.add_argument('--gallery', nargs='+', help='restrict to these corpus galleries')
    ap.add_argument('--spread', action='store_true', help='pages evenly spaced across the corpus')
    ap.add_argument('--offset', type=int, default=0, help='skip this many pages first')
    ap.add_argument('--samples', action='store_true', help='only the pages the Settings benchmark sampled')
    ap.add_argument('--cap', type=int, default=10, help='batch_size (the app sends 10 for DeepSeek, 1 for offline)')
    ap.add_argument('--capture', action='store_true', help='snapshot capture on (off in the baseline)')
    ap.add_argument('--set', action='append', help='config override, e.g. translator.translator=sugoi')
    ap.add_argument('--warm', type=int, default=0, help='warm-up pages before measuring')
    ap.add_argument('--delay', type=float, default=60, help='fair: seconds before the small job')
    ap.add_argument('--small', type=int, default=10, help='fair: pages in the small job')
    ap.add_argument('--poll', type=float, default=1.0)
    ap.add_argument('--allow-busy', action='store_true')
    args = ap.parse_args()
    result = asyncio.run(main(args))
    if args.mode == 'diff':
        sys.exit(0 if result else 1)
