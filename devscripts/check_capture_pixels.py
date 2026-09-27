"""Compare capture changes against the native renderer's own repeatability."""
import asyncio
import argparse
import copy
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from manga_translator.config import Config
from manga_translator.snapshot import Snapshot
from manga_translator.rendering.shiori_render import dispatch_shiori_render, dispatch_shiori_render_v2
from benchmark_typesetting import load_page, regions_from


async def main(args):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('manga_translator.snapshot_control', root / 'dev/performance-20260922/snapshot-before.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.ROOT = root
    rows = json.loads(Path('dev/performance-20260922/index.json').read_text())
    rows = [rows[4], rows[60], next(r for r in rows if r['gallery']=='1789561994212' and r['regions']>=5)]
    rows = [r for r in rows if not args.gallery or r['gallery'] == args.gallery]
    results = []
    for row in rows:
        m, raw, rgb, bg, bubbles, mask, render_mask = load_page(row)
        for renderer, fn in [('shiori', dispatch_shiori_render), ('shiori_v2', dispatch_shiori_render_v2)]:
            if args.renderer and renderer != args.renderer:
                continue
            outputs, layouts = [], []
            for cls in (old.Snapshot, old.Snapshot, Snapshot, Snapshot):
                cfg = Config(**m['config']); cfg.render.renderer = renderer
                regions = regions_from(m)
                image = Image.open(io.BytesIO(raw)); image._snapshot_original_bytes = raw
                ctx = SimpleNamespace(input=image, img_rgb=rgb, img_inpainted=bg.copy(), img_alpha=None, render_mask=render_mask, text_regions=regions)
                snap = cls(image, cfg)
                for key in ('ocr_lines','groups','grouping_input','ocr_captured','groups_captured','models'):
                    if key in m: snap.manifest[key] = copy.deepcopy(m[key])
                snap.before_render(ctx)
                out = await fn(ctx.img_inpainted, rgb, regions, bubbles=bubbles, text_mask=mask, snapshot=snap, render_config=cfg.render)
                ctx.img_rendered = out; ctx.result = Image.fromarray(out); snap.finish(ctx)
                outputs.append(out)
                layouts.append(sorted(snap.manifest['render_calls'][0]['engine_layout'], key=lambda info: info['nodeId']))
            comparison = []
            for a,b,label in [(0,1,'before-repeat'),(2,3,'after-repeat'),(1,2,'before-after')]:
                delta = np.abs(outputs[a].astype(np.int16)-outputs[b].astype(np.int16))
                changed = np.any(delta, axis=2)
                comparison.append(dict(kind=label, changed_pixels=int(changed.sum()),
                    changed_percent=float(changed.mean()*100), mean_absolute_channel_error=float(delta.mean()),
                    max_channel_error=int(delta.max()), layout_equal=layouts[a]==layouts[b]))
            changed_fields = sorted({field for a,b in [(0,1),(2,3),(1,2)]
                                     for one,two in zip(layouts[a],layouts[b])
                                     for field in one if one[field]!=two[field]})
            results.append(dict(gallery=row['gallery'],page=row['page'],renderer=renderer,comparisons=comparison,
                                changed_layout_fields=changed_fields,
                                changed_layout_values=[{'nodeId':one['nodeId'], 'field':field, 'before':one[field], 'after':two[field]}
                                    for one,two in zip(layouts[1],layouts[2]) for field in one if one[field]!=two[field]]))
    Path(args.output).write_text(json.dumps(results,indent=2))
    print(json.dumps(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery')
    parser.add_argument('--renderer')
    parser.add_argument('--output', default='dev/performance-20260922/native-repeatability.json')
    asyncio.run(main(parser.parse_args()))
