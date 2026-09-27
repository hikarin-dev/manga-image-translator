# Page pipeline data

The translation server keeps nothing between jobs. Every translated page comes back with its
**pipeline data**: what each stage produced, compact enough for the client to store with the
page. A later translation sends that data back with the stage to run from, and the server
restores everything before it instead of running it. The client decides what can be skipped;
the server only reports the facts it needs (below) and checks what it is given.

Code: `manga_translator/page_data.py` (format), `manga_translator/page_reuse.py` (restore and
capture in the pipeline), `manga_translator/stages.py` (config ownership and builds).

## Stages

`prepare → detect → ocr → merge → translate → mask → inpaint → bubbles → render`

Kept per page: detection lines and raw text mask, OCR text, merged regions, translations,
the refined text mask, balloon masks. `prepare`, `inpaint` and `render` always run (inpainting
is fast; its lossless background is not worth storing).

## Deciding what to run: `POST /translate/gallery/resolve`

Form field `config` → JSON:

```json
{ "config": { …effective config, only fields some stage reads… },
  "builds": { "prepare": "738fd8c45856", "detect": "…", …, "render": "…" },
  "fields": { "detect": ["detector.detector", "detector.detection_size", …], … } }
```

A **build** (12 hex) identifies the code, models and server parameters behind a stage's output;
builds are computed when the server starts, from the code itself — never bumped by hand. What each
stage runs is listed in `stages.py`, and `test/test_stage_coverage.py` fails on pipeline code that
runs without being listed (see `AGENTS.md`). A stage without an entry does not run for that
config (`bubbles` for renderers that do not read balloons). A client compares, stage by stage in
order, the build and the values of that stage's `fields` with what produced its saved data; the
first difference is where a run starts.

## Record

```json
{ "lines":   [ { "pts": [[x,y],[x,y],[x,y],[x,y]], "score": 0.93,
                 "text": "…", "prob": 0.99, "fg": [0,0,0], "bg": [255,255,255], "dir": "v" } ],
  "read":    [2, 0, 1],
  "regions": [ { "lines": [2, 0], "size": 29, "prob": 0.99, "fg": [0,1,0], "bg": [0,1,0],
                 "lang": "ja", "src": "v", "panel": 0, "tr": "Good morning!" } ],
  "bubbles": [ { "box": [700, 90, 240, 380], "bits": "…" } ] }
```

- `lines`: detected lines in detection order (`pts`, `score`), plus OCR's output for the lines
  it recognized. `read`: OCR's order (lines missing from it were dropped). Regions reference
  lines by index, so geometry and source text are stored once. These indices are also the region
  and line ids of study bubbles and feedback reports.
- Optional: `end` (a page that stopped early — the stage that found no text), region `text` (only
  when it differs from its lines joined, e.g. after the pre-dictionary), `angle` (non-zero),
  `panel` (panel-aware sorting), `keep: false` (filtered out after translation).
- Values restore with the exact types the pipeline produces (`test/test_page_data.py` round-trips
  real pages from three OCR models).

## Container

Request `stage` files and status-9 result frames carry
`[u32 big-endian JSON length][JSON, UTF-8][blobs…]`. The JSON is the record plus
`blobs: {"raw": n, "text": n}`: the sizes of the lossless masks that follow, in that order
(raw detector text mask, refined text mask) — WebP, or PNG for a page past WebP's 16383-pixel
limit; readers go by the bytes. A request adds:

- `from`: the first stage to run (`detect` … `render`); stages before it are restored.
- `keep`: later stages among `translate`, `mask`, `bubbles` whose earlier output may be kept when
  their inputs come out identical — merged texts for translation, surviving regions (with
  detection reused) for the mask, the page itself for balloons.

## Jobs

`POST /translate/gallery/start` takes one `stage` file per image (empty for a page with no
data) and `builds`, the signature of the builds the client planned against; a server whose
builds differ answers 409 and the client plans again. Invalid page data is ignored and the page
runs in full. A worker whose own builds differ (an aux node with other packages, say) runs its
pages in full and sends no status 9, so those pages simply carry no data. A client that keeps no
page data sends `capture=0`: pages still restore from any `stage` files sent, but the worker
neither encodes nor returns their new data (no status 9).

A job that starts mid-gallery (a single page translated on its own) may add `context`: the pages
before its first image, oldest first, as `[{"src": [...], "tr": [...]}]` (at most 16 pages,
64 KB). A context-aware translator starts from them as if they had been translated earlier in the
same job; other translators ignore it.

Each finished page produces, in order: status 9 (its pipeline data, unless `capture=0`),
status 5 (the image), status 6 (study payload, when enabled). All share the envelope
`tokenLen(1) + job token + page index (4 BE) + payload`.

With study enabled, when the render is the page's output (no transparency, no resize after
rendering), status 5 carries no image and the page is its status-6 study data: the bg (WebP q95)
with each balloon's lossless text layer drawn over it (`text_and_image`, which rebuilds the page
exactly — unless the layer partition had to approximate, then the image is sent), or with the
translations typeset over it as text by the client (`text_only`, which sends no text layers).
