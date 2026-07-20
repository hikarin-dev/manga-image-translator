# Shiori translation server — setup notes

This local install powers the **Translate** button in the Shiori extension.
To expose it remotely over HTTPS for other people, see **`REMOTE-SETUP.md`**.

## Start the server
Double-click **`start-translator.bat`** (or run it from a terminal). Leave the window
open while translating; close it / Ctrl+C to stop. First launch after a reboot takes a
few seconds to load models onto the GPU.

**Stop the server:** close its console window (or Ctrl+C). If it was ever left running
without a window (orphaned process), run **`stop-translator.bat`** to free ports 5003/5004.

- API: `http://127.0.0.1:5003`  (set this in Shiori → Settings → Translation)
- GPU worker: auto-spawned on `127.0.0.1:5004`
- Runs on the RTX 3070 via CUDA (torch cu124).

## Shiori settings
- **Translation server:** `http://127.0.0.1:5003`
- **Translator:** `Sugoi` (offline, Japanese→English, no API key) is fast and good for
  manga. For Korean/Chinese pages choose `M2M100`/`NLLB`. For more natural,
  context-aware output choose **`Qwen2.5-7B — local LLM`** (see below).
- **Translate to:** English.

## Optional: local LLM translator (Qwen2.5-7B via Ollama)
A local LLM gives more natural translation than Sugoi. **Ollama** is installed
(auto-runs at login) with `qwen2.5:7b` pulled (~4.7 GB). Two Shiori translator options
both run on it — switch from the dropdown, no restart:

- **"Qwen2.5-7B — local LLM (fast, per-page)"** = the `custom_openai` slot. Each page is
  translated on its own (no cross-page memory). Runs 2 pages at a time.
- **"Qwen2.5-7B + cross-page context (slower, most consistent)"** = the `chatgpt` slot,
  pointed at Ollama. Feeds the previous few pages to the model for consistent names /
  pronouns / tone across a chapter. Runs pages one-at-a-time in order, and Shiori calls
  `/reset-context` at the start of each gallery so titles don't bleed into each other.

`start-translator.bat` sets all of this:
- `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` — **required**, or the LLM translators crash
  trying to log Japanese on the Windows console.
- `CUSTOM_OPENAI_MODEL=qwen2.5:7b` (custom_openai slot).
- `OPENAI_API_BASE=http://localhost:11434/v1`, `OPENAI_API_KEY=ollama`,
  `OPENAI_MODEL=qwen2.5:7b` (chatgpt slot → Ollama).
- `--context-size 4` — pages of history fed to the context mode.

Notes:
- Requires Ollama running. On 8 GB VRAM the model co-fits with the vision models (Q4);
  Ollama frees it from VRAM ~5 min after last use. Swap models with `ollama pull <name>`
  and update the env var(s).
- Local server tweaks for this: `server/args.py` + `server/main.py` forward `--context-size`
  and add a `/reset-context` endpoint; `manga_translator.py` adds `reset_page_context()`.

## Optional: Gemini (cloud) translator
Smarter than local Qwen, but cloud-based and rate-limited. Choose **"Gemini Flash"** in
Shiori's translator dropdown.
1. Get a free key at https://aistudio.google.com/apikey (no card needed).
2. Paste it into `.env` here as `GEMINI_API_KEY=...`.
3. Run `list-gemini-models.bat`, then set `GEMINI_MODEL` in `.env` to one of the listed
   ids (without the `models/` prefix), e.g. a Gemini Flash variant.
4. Restart `start-translator.bat`. Pick "Gemini Flash" in Shiori.

Caveats:
- The build's gemini translator already sets all safety filters to `BLOCK_NONE`, so adult
  content works; sexual content involving minors is blocked by Google regardless.
- **Free-tier rate limit varies a lot by model.** `gemini-3.5-flash` is only **5 req/min**
  (very slow); `gemini-2.5-flash` ≈ 15/min; a **flash-lite** is usually ≈ 30/min. For free-tier
  use, prefer a flash-lite or 2.5-flash over the newest model. Daily cap ≈ 1,500. Shiori runs
  Gemini one page at a time.
- Patched `gemini.py`: it crashed on rate-limit errors (`server_error_attempt` used before
  assignment); now it backs off 10s and retries instead of failing the page.
- **Privacy:** on the free tier Google may use your content to improve their products.
  Paid tier doesn't. Local Qwen sends nothing off-machine.
- Gemini runs per-page (no cross-page context — that's the `chatgpt`/Ollama mode only).

## Optional: OneOCR (Windows system OCR engine)

Two new pipeline options backed by the OCR engine that ships with the Windows 11
Snipping Tool (`oneocr.dll`, CPU-only, ~0.1–0.5 s per page, no GPU use):

- **Detector `oneocr`** — one full-page pass that both detects and reads text
  (lines come back with text, per-word boxes for the mask, and a
  printed/handwritten style flag). Pair it with **OCR `oneocr`**, which then
  passes the text through instead of re-reading, only attaching a fg/bg color
  estimate. Detection size / box threshold / unclip ratio are ignored in this
  mode; text threshold filters lines by mean word confidence.
- **OCR `oneocr` with any other detector** — the usual flow: each detected
  region is unwarped to a horizontal strip and read individually (vertical text
  included), with style + color estimate attached.

Files: on first use the three runtime files (`oneocr.dll`, `oneocr.onemodel`,
`onnxruntime.dll`) are copied automatically from the installed Snipping Tool
package into `models/oneocr/`. If that fails (Snipping Tool missing), copy them
there manually from `C:\Program Files\WindowsApps\Microsoft.ScreenSketch_*\SnippingTool\`,
or set `ONEOCR_DIR` to a folder that has them. Restart the server after first
setup. Implementation: `manga_translator/utils/oneocr.py` (ctypes binding),
`detection/oneocr.py`, `ocr/model_oneocr.py`.

Caveat: oneocr.dll loads its own bundled `onnxruntime.dll`. The pip
`onnxruntime` package (used only by `inpainting/booru_tagger.py`) may clash if
both end up loaded in one process — not an issue with the default pipeline.

## OCR `mocr_fast` (manga-ocr, batched)

Same text output as `mocr` (verified: 0 mismatches over a 47-region test set) at
~4.6× the speed (1.98 → 0.43 s/page on the 3070). Two changes vs stock `mocr`:
manga-ocr runs in batches of 16 instead of once per region, and the
probability/font colors come from the 48px **CTC** model's trained color head
(one batched forward) instead of the autoregressive 48px beam decode that
dominated stock runtime. Colors are trained-model estimates either way; they
agree with stock within a mean max-channel delta of ~5 (fg) / ~7 (bg).
No torch.compile involved, so no per-restart JIT warmup (unlike `48px_exp`).
First use downloads `ocr-ctc.zip` into `models/ocr/`. Implementation:
`manga_translator/ocr/model_manga_ocr_fast.py`.

## OCR `mocr_tflite` (manga-ocr, LiteRT/TFLite, CPU)

The KV-cached LiteRT conversion of manga-ocr from HuggingFace
`jgalamba/manga-ocr-kvcache-tflite`: int8 dynamic-range ViT encoder +
fp16 2-layer decoder with explicit `init`/`step` KV-cache signatures,
greedy-decoded via `ai-edge-litert` (in requirements.txt). Runs on **CPU**
(XNNPACK), leaving the GPU free for detection/inpainting — but it is slower
per crop than GPU `mocr_fast` (~0.47 s/crop CPU vs ~0.11 s/crop CUDA on this
machine), so it's an experiment/fallback slot, not the daily default.
On a 4-crop synthetic test its text output was identical to torch manga-ocr;
upstream reports 88.4% char accuracy (CER 0.116) from the int8/fp16
quantization, so expect occasional divergence on hard crops.
Region handling and prob/font colors are inherited from `mocr_fast` (48px CTC
color head, GPU). First use downloads ~140 MB (`encoder_int8.tflite`,
`decoder_cache_fp16.tflite`, `mocr2025_vocab.csv`) into `models/ocr/`,
sha256-verified. Implementation: `manga_translator/ocr/model_manga_ocr_tflite.py`.

## Balloon-aware rendering (`manga2eng`)

The `manga2eng` renderer lays text out against a balloon mask. That mask used to
come from a Canny/contour heuristic (`rendering/ballon_extractor.py`), which
misses on borderless balloons, screentone, and balloons touching linework — and
because it was unreliable, the renderer's own overflow correction was left
disabled upstream, so long translations simply spilled outside the balloon.

The mask now comes from a YOLO11n-seg model, which makes the correction safe to
apply. Text reflows to the balloon's real width instead of overflowing in a
narrow column. Regions with no matching balloon (SFX, borderless captions, model
misses) fall back to the contour extractor's mask; the overflow-shrink correction
stays disabled for those, since that mask is often smaller than the real balloon.

Every region is typeset by **scanline polygon-fill** against its mask (segmented
or contour): each line takes the balloon's real inside-width at the rows it
occupies (so lines follow the contour — wide in the middle of an oval, narrow at
the ends, full width in a rectangular caption box), the block is bounded to the
balloon vertically, and a binary search picks the largest font whose text fully
fits. Two things make the text *fill the geometry* rather than sit as a small
rectangle: the font may grow past the source size into a roomy balloon (up to
`bubble_fill_upscale`, default 1.4×, trusted masks only — the way a letterer
picks the size for the balloon), and the final line breaks come from a
Knuth-Plass-style demerit-minimising split (`_balanced_wrap`, 2026-07-19)
instead of greedy first-fit. Demerits per line: squared relative slack (each
line's length tracks the width available at its rows — short lines at an
oval's tips, long through the middle), a smoothness term between consecutive
lines' fill ratios (a short line costs once at the block's tips but twice in
its middle, keeping the silhouette convex like a letterer's diamond), an
orphan term (lone word under half its band), and per-gap break penalties —
negative after clause punctuation, positive after articles/prepositions/
auxiliaries/numerals/lone punctuation (`_GLUE_WORDS`; weights `_WRAP_*` at the
top of `text_render_eng.py`). Both the font-size search and the re-break run at
true word granularity (2026-07-19): `seg_eng` still tokenises the translation
(splitting on spaces and after punctuation) but its extra habit of *gluing*
short words into length-based chunks like "IT TO" is undone before fitting —
those chunks would bind arbitrarily across a line break and can block a font
size the search could otherwise take a notch larger (measured: identical size
on 14/15 real bubbles, one gained a size). Splitting them back only adds break
options, so the fit stays feasible, and the demerit penalties do the phrase
binding the chunks used to. The chunk list survives *only* for the centred
`layout_lines_aligncenter` fallback (used when text can't fit even at the floor,
or for untrusted contour masks), which has no demerit machinery and leans on
gluing as its sole orphan guard. Reaches manga2eng and the shiori_v2 hybrid (its
balloon regions route through this fit); shiori v1's Rust engine is untouched. The block is always centred on the
balloon's rows (2026-07-19): the fit only accepts a k-line block that actually
uses all k lines (greedy finishing early used to leave the block riding high in
a taller block — the search now settles a notch smaller instead), and the
re-break keeps the centred anchor rather than drifting up/down a half/full line
for a marginally better fill, which visibly misaligned the text. A fitted block
is pasted exactly where it was measured (no re-centring on the crop bbox, which
used to displace it off the measured rows and clip). This replaced the old
estimate-and-shrink, which
fit a roughly rectangular centred block and routinely overflowed. When text
genuinely can't fit at a readable size it falls back to the old centred layout
(flagged `FLOOR` in `bubbles.png`).

Padding inside the balloon is `bubble_padding_ratio` (default 0.06 of the
balloon's smaller dimension) in `render_textblock_list_eng`; the wrap keeps text
inside the balloon eroded by that margin. The erosion is border-aware (the crop
is padded with background first): flat balloon edges lying on the crop border —
rectangular caption boxes, straight balloon sides — used to erode from the wrong
side and got zero padding. Under `--verbose`, `bubbles.png` is
drawn over the **rendered** page (translated text visible) and shows each
balloon, its padded inner boundary (thin line), the drawn text rect, font px and
fit %. In gallery runs the dump is routed through each page's own result folder,
so every rendered page gets its `bubbles.png`.

**Requires a one-time model export** (~11 MB) — without it the code logs a warning
and silently uses the contour fallback:

```
models/bubble_seg/manga109_yolo11n_seg.onnx
```
YOLO11n-seg trained on Manga109 + MS92/MangaSegmentation, from
https://huggingface.co/huyvux3005/manga109-segmentation-bubble (`best.pt`).
The HF repo only hosts the PyTorch checkpoint; export it with ultralytics
(installed in the venv with `--no-deps` plus onnx/py-cpuinfo/ultralytics-thop/
matplotlib, so the pinned torch stays untouched):
`YOLO('best.pt').export(format='onnx', imgsz=1600, dynamic=True, simplify=False, opset=17)`.
Inference letterboxes to 1600 (the training size — `INPUT_SIZE` in
`bubble_seg.py`). Replaced kitsumed/yolov8m_seg-speech-bubble (640) on
2026-07-19; that model and its files were removed (to revert, re-download its
`model_dynamic.onnx`, point `MODEL_PATH` at it and set `INPUT_SIZE = 640`).
A/B over 136 result
pages: identical bubble totals, matched-mask IoU 0.92, the new model finds
thin sliver balloons the old one missed, and rarely merges two adjacent
narration captions into one region (which `assign_regions` safely drops to
the contour fallback).

Runs on CPU via the already-installed `onnxruntime` (~0.7 s/page), inside the
render lane. Set `MT_BUBBLE_SEG=0` to disable and restore the old behaviour.
Implementation: `manga_translator/rendering/bubble_seg.py`. Note the exported
ONNX carries Ultralytics' AGPL-3.0 in its metadata, not the Apache-2.0 on the
model card — relevant only because the server is exposed publicly.

## Shiori renderer (`render.renderer: "shiori"`) — koharu engine

A second, fully independent English renderer built on [koharu](https://github.com/mayocream/koharu)'s
text engine (GPL-3.0). Completely isolated from `text_render_eng.py` — selecting
`"shiori"` as the renderer touches none of the manga2eng code paths.

**Layout/rendering (Rust, `shiori-renderer/`):**
- `vendor/koharu-renderer/` — koharu v0.61.2's renderer crate, **byte-identical
  vendored sources** (harfrust shaping, skrifa metrics, fontdue+tiny-skia raster,
  hypher hyphenation, ICU segmentation, Knuth-Plass-style DP line breaking).
- `src/driver.rs` — faithful port of `koharu-app/src/renderer.rs` (the render
  driver: bubble-ID mask → per-block layout-box expansion, binary-search font
  fit with pixel-level mask-collision checks, stroke/text color resolution).
  Deviations are marked `[shiori]`: Google-Fonts service removed (fonts come
  from the system + registered files), `NodeId` is a caller index, fitted font
  size reported back. All 13 upstream driver tests ported and passing.
- `src/lib.rs` — PyO3 bindings (`shiori_renderer` abi3 wheel, installed in the
  venv). Rebuild: `PATH="$HOME/.cargo/bin:$PATH" ../venv/Scripts/maturin.exe
  build --release -o dist` then `pip install --force-reinstall dist/*.whl`
  (rustup lives at `~/.cargo`, installed with `--no-modify-path`).

**Colors/style (Python, independent of OCR):**
- `manga_translator/rendering/shiori_style.py` — koharu's YuzuMarker font
  detector (ResNet-50, weights auto-download from
  `fffonion/yuzumarker-font-detection` to `models/shiori/`). Per region-crop of
  the SOURCE page it regresses text RGB, stroke RGB, stroke width, direction,
  size, angle, plus koharu's normalization (near-black/white clamping; stroke
  suppressed when text≈stroke color). Faithful to koharu except: torchvision
  maxpool keeps padding=1 (candle can't pad; the weights were trained with it)
  and inference is fp32.
- Color policy (koharu's, in the driver): predicted text color used directly;
  stroke uses the predicted width but an auto-contrast black/white color.
- `manga_translator/rendering/shiori_render.py` — glue: detection boxes as seed
  transforms, `bubble_seg` masks → grayscale bubble-ID mask, predictions per
  block, CC Victory Speech registered as the document font.

Select with `render.renderer: "shiori"` in the request config. Blocks keep
koharu semantics: text expands into its balloon (single-tenant balloons only),
centre-aligned, hyphenated, capped at 72 px / floored at a size derived from
page dimensions.

The driver reports the colors each block was ACTUALLY drawn with (text +
resolved stroke; stroke color == text color and width 0 when no stroke was
drawn). The glue stores them as `region._drawn_fg/_drawn_bg`, and the study
payload's single `style.fg`/`style.bg` pair prefers them over OCR colors —
both the original and the translation DOM text share that one pair, so study
text always matches the render.

## Post-OCR color override (`render.estimate_font_color` / `estimate_outline_color`)

Optional per-request flags (Shiori: two checkboxes under Settings → Translation
→ OCR) that re-estimate text colors from the image after any OCR model runs
and overwrite the fill color and/or outline color independently. The estimator
(`manga_translator/ocr/colors.py`) uses a detector glyph mask when available,
sampling the glyph core for fill and its narrow boundary for a real outline. If
there is no distinct outline it samples only the first two pixels outside the
glyph, producing a local background-colored (invisible) outline without being
pulled toward distant artwork. Filled detector boxes are rejected and fall back
to image-only Otsu segmentation. It then clusters the page's colors (max channel
diff ≤ 40) and snaps each cluster to its darkest member (dark clusters) or
brightest (light clusters) — so a page of near-black dialogue renders one solid
black instead of per-bubble grey jitter. Chromatic fill clusters retain their
most saturated member, while outline clusters use the light/dark endpoint so
pale antialias pixels cannot tint a white outline. Fill-to-white/black blends
are extrapolated and clamped to the exact endpoint; a clear same-page white
consensus among pink text also repairs isolated mask outliers. Forced
`font_color` still wins over the estimate.

## Local modifications (don't `git pull` over these without re-checking)
- `manga_translator/textline_merge/__init__.py`: a merged text block's font colors
  are the **dominant color by text area** among its lines instead of the mean —
  blocks that merge lines of different colors (another speaker, colored emphasis)
  no longer blend into a muddy dark mix. The post-OCR color override also snaps
  chromatic fill clusters to their most saturated member (`ocr/colors.py`) so
  colored fill text stays vivid; outline clusters deliberately do not use that
  rule.
- `requirements.txt`: `transformers` pinned to `4.46.3` (5.x needs torch ≥2.7 and breaks
  manga-ocr); `pydensecrf` commented out (needs a C++ compiler that isn't installed).
- `manga_translator/mask_refinement/text_mask_utils.py`: `refine_mask` imports pydensecrf
  lazily and falls back to the raw mask when it's absent.
- `start-translator.bat`: sets `MT_WEB_NONCE=None` — the server never forwards the
  internal nonce to its worker, so leaving it on causes a 401 "Nonce does not match".
- `fonts/ccvictoryspeech.ttf` + `manga_translator/rendering/__init__.py`: the `manga2eng`
  renderer's default font is set to CC Victory Speech (converted from the Ichigo extension's
  bundled woff2) for Ichigo-style comic lettering. To revert, point it back at
  `fonts/comic shanns 2.ttf`. After changing the renderer code, restart the server.
- `manga_translator/rendering/text_render_eng.py`: takes its balloon mask from
  `rendering/bubble_seg.py` when a balloon matches the text region, and re-enables the
  upstream-disabled overflow correction for those regions only. See "Balloon-aware
  rendering" above.

## Python env
Dedicated venv at `.\venv` (Python 3.11.9). To reinstall deps:
`venv\Scripts\python.exe -m pip install -r requirements.txt`
