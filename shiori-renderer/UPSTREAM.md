# Shiori renderer: upstream port

The `shiori` renderer uses Koharu's scene renderer and Vello rasterizer from
[`4a133539f204ab1182ff64901ba5226b4e868fb0`](https://github.com/koharu-rs/koharu/tree/4a133539f204ab1182ff64901ba5226b4e868fb0)
(upstream 0.83.4, 2026-09-18). The commit is the source identity; existing Shiori
package version fields were deliberately not bumped during development.

This replaces the previous 0.61.2 driver, TinySkia rasterization, byte-valued
bubble-ID map, size heuristics, and YuzuMarker style model. The app option remains
`render.renderer: "shiori"` and is displayed as **shiori**.

## What is upstream

Six crates are vendored from that revision: `koharu-renderer`, `koharu-rasterizer`,
`koharu-scene`, `koharu-storage`, `koharu-config`, and `koharu-runtime`.
`Cargo.lock` pins their dependency graph, including the `hf-hub` Git revision.
Upstream code is MIT OR Apache-2.0; both license texts are in `vendor/`.
The Python integration retains its existing GPL-3.0-only license.

The actual upstream `Renderer::render` processes a `koharu-scene::Snapshot`.
It owns contour fitting, physical lobe division for conjoined balloons, source
anchor assignment, automatic size search, readable-size diagnostics, paragraph
and point text, explicit line breaks, last-resort language-aware hyphenation,
ICU/Jieba segmentation, CJK punctuation and vertical layout, RTL shaping, system
font fallback, variable faces, synthetic bold/italic, alignment, rotation, fill,
outlines, and compositing. The unchanged rasterizer handles GPU rendering,
antialiasing, surface limits, and optional supersampling.

Automatic text without a balloon retains upstream's **24 px maximum**. Balloon
text uses the upstream contour capacity and readable-size policy. Shiori does
not add an ellipse, collision shrink loop, or larger free-text size cap.

`src/typography.rs` copies upstream `koharu-pipeline/src/stages/detection.rs`
typography inference, including fill/outline separation, measured stroke width,
angle, and source direction. Only input container types, visibility, and output
serialization differ. Its 13 upstream behavioral tests are also included.
`driver::mask_geometry` uses that module's contour extraction formula.

## Host integration and local patches

`src/driver.rs` translates pipeline inputs into scene entities and relations.
Texts sharing a segmentation instance share a `FlowsIn` relation while retaining
separate source anchors; upstream splits the actual joined contour. Uncontained
texts use `FitsTo`. The inpainted RGB page becomes the scene background.

The Python adapter selects the smallest bubble covering at least 90% of the
region's text pixels, preserving overlapping instances and IDs above 255.
The detector mask is restricted to the region's OCR quadrilaterals. If no detector
mask exists, quadrilaterals supply containment and OCR supplies the fallback
colors/direction. Explicit render size, direction, alignment, colors, and border
disable settings are passed through. The configured font file is registered;
changing it recreates the font context. Text wording is not rewritten here.

Only these upstream renderer files have local patches:

- `fonts.rs`: register the caller's font bytes in upstream's fontique collection;
  resolve typographic family names before legacy family names.
- `renderer.rs`: expose construction from an in-memory typesetting configuration
  and font registration; carry snapshot metadata through the frame.
- `layout.rs`, `text_renderer.rs`, `frame.rs`, `lib.rs`: expose the actual chosen
  lines, source byte ranges, inserted hyphens, baselines, advances, and loaded font
  bytes/face indices for study hints and feedback snapshots. These observations
  do not participate in fitting, shaping, or rasterization.

Study hints retain `strokeWidth` as the outward outline thickness in page pixels,
including zero. The reader doubles positive values for its centred CSS stroke and
scales them with the page. `borderDisabled` and `strokeColorExplicit` distinguish
manual settings from inferred paint and historical no-outline records.

### Shiori black-text outline policy

At the user's request, black/near-black fill (all RGB channels at most 32) receives
a white outline at least 1.5 page pixels thick, even when source inference found
none. A larger inferred width is retained. This policy lives in the Python adapter;
explicit outline color and border-disable settings override it. The reader applies
the same rule to selectable source/translation text, including historical records.
White and colored fills retain their inferred outline behavior.

This is an intentional paint-policy difference from upstream, whose
`resolve_stroke` requires positive width and otherwise draws no outline. Koharu's
fitting, line breaking, shaping, and rasterizer code remain unchanged. Snapshot
block inputs and layout hints record the actual white outline so replay stays
faithful. The native parity checker still compares identical supplied inputs;
Shiori's default black-text paint inputs now differ from raw upstream inference.

### Shiori hybrid paint

The `shiori_v2` adapter uses this same scene renderer for all regions, including
joined balloons and free text. It uses manga2eng's `TextBlock.get_font_colors()`
and shared FreeType stroke-radius formula instead of upstream's inferred paint.
Inference still supplies angle and source direction. The native `layout_page`
binding resolves the chosen font size without rasterizing; Python then sets the
actual outward radius and renders with unchanged automatic fitting inputs.
The radius is `max(int(0.07 * int(size)), 1)` when manga2eng's 10% integer border
gate is positive, otherwise zero. Explicit border disable forces zero.

This changes paint inputs only. Rasterization remains Vello, so glyph-edge
antialiasing is not claimed to match manga2eng's FreeType pixels. Snapshot block
inputs contain the resolved widths, and `paint_policy: "manga2eng"` identifies the
hybrid. Study `paintPolicy` preserves those colors and widths without applying
the ordinary Shiori black-text halo or manga2eng's uppercase layout convention.

All other vendored source files are unchanged. There is no old-renderer fallback.
The native module exposes `UPSTREAM_REVISION`; the adapter rejects an incompatible
installed wheel instead of claiming to use the new implementation.

## Parity boundary

Equal wording, geometry, typography, font bytes/fallbacks, page pixels, and raster
settings produce the upstream renderer's result. This is a renderer port, not a
replacement of Shiori's OCR, text grouping, bubble segmentation, translation, or
inpainting models. Their inputs can differ from Koharu's RF-DETR pipeline, so a
whole page processed independently in the two applications can still differ.
Selecting a different document font also changes layout. Machine-specific system
fallback fonts and GPU drivers can affect pixels.

`tools/check_parity.py` builds a separate reference executable against pristine
upstream crates, with no capture patches. Only scene construction is shared.
It checks 16 fixtures: oval and two-/three-lobe balloons, free text, hyphenation,
newlines, colored outlines/bold/italic, rotation, Japanese vertical, Chinese,
Arabic RTL, mixed-script fallback, explicit size/alignment, point-text clipping,
supersampling, and empty translations. Both engines use Arial and system fallback.
It saves both PNGs, metadata, and raw pixel-error counts. Layout fields are compared
at native float32 precision (JSON serializers may print different decimal lengths).

The final installed release wheel matched all 16 fixtures byte for byte, with
matching layout metadata. An initial debug comparison matched 15 fixtures byte
for byte; the remaining
styled-outline fixture differed at one pixel by one channel level. Six repeat
runs reproduced that same variation in **pristine upstream versus itself** and
in the port versus itself. The checker therefore allows at most one differing
pixel with a maximum channel error of one, reports the raw counts, and requires
matching layout metadata. It does not claim universal bit-exact GPU output.

A separate hybrid check over gray backgrounds found a larger repeatability limit:
even pristine upstream repeated on this GPU varies in faint white-outline fringe
pixels (21–31 pixels, up to 30 channel levels, for a 280×180 Arial fixture).
Solid fill/outline pixels and layout remained equal. Hybrid regressions therefore
check exact layout, dark fill (including its antialiasing), and solid outlines;
they do not assert repeatability of the faint outer white fringe.
The original 16-fixture upstream parity limits are unchanged.

## Build and validate

Use Rust with MSVC build tools on Windows, Python 3.9+, and a GPU/driver supported
by upstream wgpu/Vello. The renderer needs this GPU support even if OCR uses CPU.
The configured local font and system fallback work without model downloads.
Upstream's optional bundled-font catalogue/download support is retained.

From the backend root in PowerShell:

```powershell
$env:PATH = "$env:USERPROFILE/.cargo/bin;" + $env:PATH
& venv/Scripts/python.exe -m pip install maturin
Push-Location shiori-renderer
& ../venv/Scripts/maturin.exe build --release --locked -o dist
Pop-Location
& venv/Scripts/python.exe -m pip install --force-reinstall --no-deps shiori-renderer/dist/shiori_renderer-0.1.0-cp39-abi3-win_amd64.whl
& venv/Scripts/python.exe -m pytest test/test_shiori_render.py test/test_snapshots.py -q
cargo test --manifest-path shiori-renderer/Cargo.toml -p koharu-renderer -p shiori-renderer --lib
```

Restart workers that loaded the previous extension, and reload the app.
`requirements.txt` no longer installs the incompatible old release wheel. Install
this optional renderer separately from source using the commands above (or
`pip install ./shiori-renderer` from the backend root). This leaves ordinary
requirements-only installations usable without Rust when they select another
renderer. Rust dependencies must be fetched on first build; subsequent builds
can use `--offline`.

To reproduce the pixel comparison, clone upstream, check out the exact commit
above, then run:

```powershell
& venv/Scripts/python.exe shiori-renderer/tools/check_parity.py --upstream dev/koharu --output dev/koharu-parity
```

The checkout's crates must be pristine. Saved fixtures and outputs stay under the
chosen output directory. Reusing that directory refreshes its comparison files.
