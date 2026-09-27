"""Replay real saved page inputs, without changing library records or calling an LLM.

Run from the backend root. The index is built by matching browser-cached original
SHA-256 values to snapshot manifests. Segmentation/translation are held constant;
these are render-and-capture service times, not whole-pipeline throughput. This
control uses a normalized RGB result. The production dump_image composition into
RGBA, model work, study overlays and transport are measured by the HTTP harness.
"""
import argparse
import asyncio
import copy
import hashlib
import io
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from manga_translator.config import Config
# Retain a runnable control after production optimizations are edited. This is
# deliberately a benchmark-only switch, never a pipeline configuration option.
if os.environ.get('TYPESETTING_BENCHMARK_BASELINE'):
    root = Path(__file__).resolve().parents[1]
    for name, filename in (('manga_translator.snapshot', 'snapshot-before.py'),
                           ('server.snapshots', 'storage-before.py')):
        spec = importlib.util.spec_from_file_location(name, root / 'dev/performance-20260922' / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.ROOT = root if name.startswith('manga_translator.') else root / 'snapshots'
from manga_translator.snapshot import Snapshot, restore_state, region_state
from manga_translator.utils import TextBlock
from manga_translator.rendering import dispatch_eng_render
from manga_translator.rendering.shiori_render import dispatch_shiori_render, dispatch_shiori_render_v2
from server.snapshots import save_archive


class TimedSnapshot(Snapshot):
    def __init__(self, *args):
        self.timings = defaultdict(float)
        super().__init__(*args)


def instrument(name):
    original = getattr(Snapshot, name)
    def measured(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            self.timings[name] += time.perf_counter() - start
    setattr(TimedSnapshot, name, measured)


for method in ('image', 'asset', 'begin_call', 'before_render', 'finish', 'model_versions'):
    instrument(method)


def load_page(row):
    manifest = json.loads(Path(row['manifest']).read_text('utf-8'))
    def raw(ref):
        return (Path('snapshots/objects') / ref['sha256']).read_bytes()
    def image(ref):
        return np.array(Image.open(io.BytesIO(raw(ref)))) if ref else None
    original = raw(manifest['original'])
    processed, background = image(manifest['processed']), image(manifest['background'])
    calls = [c for c in manifest['render_calls'] if c.get('regions')]
    call = calls[0] if calls else {}
    bubbles = [image(ref) for ref in (call.get('page_bubble') or call.get('bubble_masks') or [])]
    # Native snapshots preserve detector pixels inside each OCR region. Older
    # manga2eng captures lack these: explicitly use the same OCR-quad fallback
    # for both native renderers on those pages, never invent detector evidence.
    text_mask = None
    if call.get('typography_text_masks'):
        text_mask = np.zeros(processed.shape[:2], np.uint8)
        for item in call['typography_text_masks']:
            if not item['mask']:
                continue
            mask = image(item['mask'])
            x, y = item['x'], item['y']
            text_mask[y:y+mask.shape[0], x:x+mask.shape[1]] |= mask
    return manifest, original, processed, background, bubbles, text_mask, image(manifest.get('render_mask'))


def regions_from(manifest):
    regions = []
    for item in manifest['render_regions']:
        r = TextBlock.__new__(TextBlock)
        r.__dict__.update(restore_state(item['state']))
        r._snapshot_region_id = item['id']
        regions.append(r)
    return regions


async def measure(row, renderer, capture, loaded, storage):
    m, raw, rgb, bg, bubbles, text_mask, render_mask = loaded
    config = Config(**m['config'])
    config.render.renderer = renderer
    regions = regions_from(m)
    input_image = Image.open(io.BytesIO(raw))
    input_image._snapshot_original_bytes = raw
    ctx = SimpleNamespace(input=input_image, img_rgb=rgb, img_inpainted=bg.copy(),
                          img_alpha=None, render_mask=render_mask, text_regions=regions)
    start = time.perf_counter()
    snapshot = TimedSnapshot(input_image, config) if capture else None
    if snapshot:
        for key in ('ocr_lines', 'groups', 'grouping_input', 'models', 'ocr_captured', 'groups_captured'):
            if key in m:
                snapshot.manifest[key] = copy.deepcopy(m[key])
        if regions:
            snapshot.before_render(ctx)
    render_start = time.perf_counter()
    if renderer == 'manga2eng':
        out = await dispatch_eng_render(ctx.img_inpainted, rgb, regions,
            line_spacing=config.render.line_spacing, disable_font_border=config.render.disable_font_border,
            page_bubbles=bubbles, safe_layout=True, snapshot=snapshot)
    else:
        fn = dispatch_shiori_render if renderer == 'shiori' else dispatch_shiori_render_v2
        out = await fn(ctx.img_inpainted, rgb, regions, bubbles=bubbles,
                       snapshot=snapshot, render_config=config.render, text_mask=text_mask)
    render_s = time.perf_counter() - render_start
    ctx.img_rendered = out if regions else None
    ctx.result = Image.fromarray(out)
    if snapshot:
        snapshot.finish(ctx)
    capture_end = time.perf_counter()
    persist_s = 0
    if snapshot:
        t = time.perf_counter()
        save_archive(ctx.result._snapshot_archive, storage)
        persist_s = time.perf_counter() - t
    layout = [{k: region_state(r).get(k) for k in ('_drawn_lines', '_drawn_font_size', '_drawn_rect')} for r in regions]
    return dict(gallery=row['gallery'], page=row['page'], renderer=renderer, capture=capture,
                regions=len(regions), size=m['processed']['size'], render_s=render_s,
                total_s=capture_end-start+persist_s, persist_s=persist_s,
                timings=dict(snapshot.timings) if snapshot else {},
                archive_bytes=len(ctx.result._snapshot_archive) if snapshot else 0,
                pixels_sha256=hashlib.sha256(out.tobytes()).hexdigest(), layout=layout,
                text_mask='detector-within-quads' if text_mask is not None else 'quad-fallback')


async def main(args):
    rows = json.loads(Path(args.index).read_text())
    if args.limit:
        rows = rows[:args.limit]
    if args.empty_only:
        rows = [r for r in rows if not r['regions']]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if target.exists():
        existing = {(r['gallery'], r['page'], r['renderer'], r['capture'])
                    for r in map(json.loads, target.read_text().splitlines())}
    # Separate cold model/font/native-device initialization from warm timings.
    warm_row = next((r for r in rows if r['regions']), None)
    if warm_row:
        loaded = load_page(warm_row)
        for renderer in args.renderers:
            t = time.perf_counter()
            await measure(warm_row, renderer, False, loaded, None)
            print(json.dumps({'warmup': renderer, 'seconds': time.perf_counter()-t}), flush=True)
    with target.open('a', encoding='utf-8') as stream:
        for i, row in enumerate(rows):
            loaded = load_page(row)
            renderers = args.renderers[i % len(args.renderers):] + args.renderers[:i % len(args.renderers)]
            for renderer in renderers:
                for capture in ([False, True] if i % 2 == 0 else [True, False]):
                    key = row['gallery'], row['page'], renderer, capture
                    if key in existing:
                        continue
                    result = await measure(row, renderer, capture, loaded, target.parent / (target.stem + '-storage'))
                    stream.write(json.dumps(result) + '\n')
                    stream.flush()
            print(json.dumps({'completed': i+1, 'total': len(rows), 'gallery': row['gallery'], 'page': row['page']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', default='dev/performance-20260922/index.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--empty-only', action='store_true')
    parser.add_argument('--renderers', nargs='+', default=['manga2eng', 'shiori', 'shiori_v2'])
    asyncio.run(main(parser.parse_args()))
