"""Compare the Python extension with pristine, commit-pinned upstream crates.

Run with this backend's Python, an upstream checkout, and an output directory.
Uses installed Arial (and upstream system fallback) in both engines. No models or
font downloads are needed. The shared adapter only constructs the input scene;
all reference layout, shaping and rasterization code comes from the checkout.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import shiori_renderer

PIN = '4a133539f204ab1182ff64901ba5226b4e868fb0'
ROOT = Path(__file__).resolve().parents[1]


def ellipse(cx, cy, rx, ry, count=96):
    return [{'x': cx + rx * math.cos(i * math.tau / count),
             'y': cy + ry * math.sin(i * math.tau / count)} for i in range(count)]


def fixtures():
    import cv2
    renderer = shiori_renderer.PageRenderer(['Arial'])

    def joined(circles):
        mask = np.zeros((480, 640), np.uint8)
        for x, y, rx, ry in circles:
            cv2.ellipse(mask, (x, y), (rx, ry), 0, 0, 360, 1, -1)
        return json.loads(renderer.bubble_geometry(mask.tobytes(), 640, 480))

    def block(text, x=80, y=80, width=150, height=200, **kw):
        return dict(nodeId=0, transform=dict(x=x, y=y, width=width, height=height),
                    translation=text, sourceText='source', **kw)

    def case(name, blocks, bubbles=(), language='en', supersampling=1):
        blocks = copy.deepcopy(blocks)
        for i, item in enumerate(blocks):
            item['nodeId'] = i
        return dict(name=name, width=640, height=480, blocks=blocks,
                    options=dict(documentFont='Arial', targetLanguage=language,
                                 bubbles=[dict(id=i, points=p) for i, p in enumerate(bubbles)],
                                 supersampling=supersampling))

    oval = ellipse(225, 225, 185, 205)
    text = 'We finally found a way through. Let us see what happens next!'
    yield case('oval', [block(text, bubbleId=0)], [oval])
    yield case('joined_two', [block('There is something I need to tell you.', x=100, y=80, bubbleId=0),
                              block('I already know.', x=340, y=145, bubbleId=0)],
               [joined([(195, 210, 155, 185), (420, 250, 125, 155)])])
    yield case('joined_three', [block(t, x=x, y=y, width=65, height=80, bubbleId=0)
                                for t, x, y in [('First, listen.', 100, 90),
                                                ('Then think about what I said.', 280, 140),
                                                ('Are we ready?', 455, 250)]],
               [joined([(130, 160, 115, 140), (310, 220, 120, 150), (490, 310, 110, 130)])])
    yield case('no_bubble', [block('A short remark.', width=470, height=340)])
    yield case('hyphenation', [block('Uncharacteristically incomprehensible miscommunication.',
                                     width=70, height=100)])
    yield case('explicit_breaks', [block('Wait...\nReally?!\nYes, really.', bubbleId=0)], [oval])
    yield case('paint_and_style', [block(text, bubbleId=0, color=[30, 80, 180, 255],
                                        strokeColor=[255, 220, 90, 255], strokeWidth=2.5,
                                        fontWeight=700, fontStyle='italic')], [oval])
    rotated = block(text, x=130, y=90, width=300, height=200)
    rotated['transform']['rotationDeg'] = 17
    yield case('rotation', [rotated])
    yield case('japanese_vertical', [block('本当に！？そうだったのですね。', sourceDirection='vertical',
                                           bubbleId=0)], [oval], language='ja')
    yield case('chinese_horizontal', [block('我们终于找到了一条路。接下来会发生什么呢？', bubbleId=0)],
               [oval], language='zh-Hans')
    yield case('arabic_rtl', [block('مرحبا بالعالم، هذه تجربة للنص العربي.', bubbleId=0)], [oval], language='ar')
    yield case('mixed_fallback', [block('Hello 世界！ Café — Ελληνικά.', bubbleId=0)], [oval])
    yield case('explicit_size_alignment', [block('One two three four five six.', fontSize=35,
                                                 alignment='End', width=330)])
    yield case('point_text_clipped', [block('An unwrapped line crossing the page edge', x=460, y=410,
                                           width=70, height=40, pointText=True, fontSize=28)])
    yield case('supersampling', [block(text, bubbleId=0, strokeWidth=1.5,
                                       strokeColor=[255, 255, 255, 255])], [oval], supersampling=2)
    yield case('empty_text', [block('')])


REFERENCE_MAIN = r'''
use anyhow::Result;
use image::RgbaImage;
use koharu_rasterizer::{Rasterizer, RasterOptions};
use koharu_renderer::{Renderer, TypesettingConfig, LayerKind, WritingMode};
use serde::Deserialize;
mod adapter;
#[derive(Deserialize)]
struct Fixture { name: String, width: u32, height: u32, blocks: Vec<adapter::Block>, options: adapter::Options }
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let fixtures: Vec<Fixture> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let renderer = Renderer::from_config(koharu_config::Config::memory(TypesettingConfig {font_families: vec!["Arial".into()]}))?;
    let rasterizer = Rasterizer::new()?;
    for fixture in fixtures {
        let image = RgbaImage::from_pixel(fixture.width, fixture.height, image::Rgba([255; 4]));
        let (snapshot, page, ids) = adapter::page_scene(image, &fixture.blocks, &fixture.options).await?;
        let frame = renderer.render(&snapshot, page).await?;
        let raster = rasterizer.rasterize(&frame.raster_frame()?, RasterOptions::supersampled(fixture.options.supersampling.unwrap_or(1)))?;
        let root = std::path::Path::new(&args[2]);
        raster.image.save(root.join(format!("{}.png", fixture.name)))?;
        let mut metadata = Vec::new();
        for (id, entity) in ids {
            let Some(layer) = frame.layer(entity) else { continue; };
            let LayerKind::Text(meta) = layer.kind() else { continue; };
            let bounds = layer.bounds();
            metadata.push(serde_json::json!({"nodeId": id, "fontSize": meta.font_size,
                "x": bounds.x, "y": bounds.y, "width": bounds.width, "height": bounds.height,
                "rotationDeg": meta.angle_degrees, "geometry": layer.geometry().points,
                "renderedDirection": if meta.writing_mode == WritingMode::Horizontal {"horizontal"} else {"vertical"}}));
        }
        std::fs::write(root.join(format!("{}.json", fixture.name)), serde_json::to_vec_pretty(&metadata)?)?;
        println!("{}", fixture.name);
    }
    Ok(())
}
'''


def build_reference(upstream, output):
    actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != PIN:
        raise SystemExit(f'Expected upstream {PIN}; got {actual}')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--', 'crates'], text=True).strip():
        raise SystemExit('Reference checkout crates must be pristine')
    workspace = output / 'reference-workspace'
    workspace.mkdir(exist_ok=True)
    for crate in (ROOT / 'vendor').glob('koharu-*'):
        shutil.copytree(upstream / 'crates' / crate.name, workspace / 'vendor' / crate.name, dirs_exist_ok=True)
    manifest = (ROOT / 'Cargo.toml').read_text()
    start, end = manifest.index('[lib]'), manifest.index('[dependencies]')
    manifest = manifest[:start] + '[[bin]]\nname = "upstream-reference"\npath = "reference.rs"\n\n' + manifest[end:]
    manifest = manifest.replace('name = "shiori-renderer"', 'name = "upstream-reference"', 1)
    manifest = manifest.replace('[dependencies]', '[dependencies]\nkoharu-config = { workspace = true }', 1)
    manifest = '\n'.join(line for line in manifest.splitlines() if not line.startswith('pyo3 =')) + '\n'
    (workspace / 'Cargo.toml').write_text(manifest)
    shutil.copyfile(ROOT / 'Cargo.lock', workspace / 'Cargo.lock')
    # Only scene construction is shared. Remove every reference to the patched
    # snapshot API; the reference compiles entirely against untouched upstream.
    adapter = (ROOT / 'src' / 'driver.rs').read_text()
    adapter = adapter[:adapter.index('/// Resolve upstream layout')]
    adapter = adapter[:adapter.index('#[derive(Serialize)]')] + adapter[adapter.index('/// Identical contour extraction'):]
    adapter = '\n'.join(line for line in adapter.splitlines() if not line.startswith('use koharu_renderer::'))
    (workspace / 'adapter.rs').write_text(adapter, encoding='utf-8')
    (workspace / 'reference.rs').write_text(REFERENCE_MAIN, encoding='utf-8')
    subprocess.run(['cargo', 'build', '--offline', '--manifest-path', str(workspace / 'Cargo.toml'),
                    '--target-dir', str(ROOT / 'target'), '--bin', 'upstream-reference'], check=True)
    return ROOT / 'target' / 'debug' / ('upstream-reference.exe' if __import__('os').name == 'nt' else 'upstream-reference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert shiori_renderer.UPSTREAM_REVISION == PIN
    args.output.mkdir(parents=True, exist_ok=True)
    reference_dir, port_dir = args.output / 'reference', args.output / 'port'
    reference_dir.mkdir(exist_ok=True)
    port_dir.mkdir(exist_ok=True)
    cases = list(fixtures())
    fixture_path = args.output / 'fixtures.json'
    fixture_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    binary = build_reference(args.upstream.resolve(), args.output.resolve())
    subprocess.run([str(binary), str(fixture_path.resolve()), str(reference_dir.resolve())], check=True)
    renderer = shiori_renderer.PageRenderer(['Arial'])
    results = []
    for case in cases:
        w, h = case['width'], case['height']
        raw, metadata = renderer.render_page(bytes([255]) * w * h * 4, w, h,
                                            json.dumps(case['blocks']), json.dumps(case['options']))
        path = port_dir / (case['name'] + '.png')
        Image.frombytes('RGBA', (w, h), raw).save(path)
        pixels = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
        expected = np.array(Image.open(reference_dir / path.name).convert('RGBA'))
        delta = np.abs(pixels.astype(np.int16) - expected.astype(np.int16))
        expected_meta = json.loads((reference_dir / (case['name'] + '.json')).read_text())
        actual_meta = json.loads(metadata)
        (port_dir / (case['name'] + '.json')).write_text(json.dumps(actual_meta, indent=2), encoding='utf-8')
        # serde_json::Value promotes f32 to f64; the extension serializes typed
        # f32 directly. Compare those fields at their actual stored precision.
        float_fields = {'x', 'y', 'width', 'height', 'fontSize', 'rotationDeg'}
        metadata_equal = len(expected_meta) == len(actual_meta) and all(
            all((np.float32(actual[key]) == np.float32(value)) if key in float_fields
                else actual[key] == value for key, value in expected.items())
            for actual, expected in zip(actual_meta, expected_meta))
        item = dict(name=case['name'], different_pixels=int(np.count_nonzero(delta.max(axis=2))),
                    max_channel_error=int(delta.max()), metadata_equal=metadata_equal,
                    rgba_sha256=hashlib.sha256(raw).hexdigest())
        results.append(item)
        print(json.dumps(item), flush=True)
    # On this GPU Vello can vary by one channel level at one outlined pixel,
    # including pristine-upstream versus itself. Keep raw errors in the report.
    report = dict(upstream=PIN, allowed_different_pixels=1, allowed_channel_error=1, fixtures=results)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    if any(r['different_pixels'] > 1 or r['max_channel_error'] > 1 or not r['metadata_equal'] for r in results):
        raise SystemExit('Parity failed; inspect report.json and saved images')


if __name__ == '__main__':
    main()
