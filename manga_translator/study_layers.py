"""Study-payload builders, split out of manga_translator.py so the process pool can run them.

A pool worker imports the module holding its function. manga_translator.py loads torch, CUDA and
every model module; this one needs only numpy and PIL, so a worker stays a few hundred MB instead
of ~2.6 GB of committed memory. The functions are unchanged."""
import numpy as np
from PIL import Image


# ── study-layer builders (module-level so they run in the GIL-free process pool) ──────────
# text_and_image study mode partitions the final render's changed pixels among a page's bubbles
# and encodes a full-page transparent layer per bubble + the shared inpaint bg: per-bubble numpy
# fills + PNG/WebP encodes + base64. On the thread pool that Python-heavy work holds the GIL and
# serializes the whole pipeline (study_overlay measured ~40% of a dense gallery's wall, with CPU
# cores idle — the GIL-serialization tell). Running it out-of-process (like mask refinement) frees
# the GIL. These are module-level, not closures/methods, so ProcessPoolExecutor can pickle them.

def _study_norm(x1, y1, x2, y2, W, H):
    return {'x': x1 / W, 'y': y1 / H, 'w': (x2 - x1) / W, 'h': (y2 - y1) / H}


def _study_meta_bubble(info, region_xyxy, W, H):
    """Per-bubble geometry + text + style, normalized to the page. Reads no pixels."""
    dx1, dy1, dx2, dy2 = info['det']
    bx1, by1, bx2, by2 = info['rbox']
    bubble = {
        'box': _study_norm(dx1, dy1, dx2, dy2, W, H),
        'rbox': _study_norm(bx1, by1, bx2, by2, W, H),
        'region': _study_norm(*region_xyxy, W, H),
        'tr': info['tr'], 'src': info['src'], 'style': info['style'],
    }
    # Optional DOM-text metadata: per-source-line ruby segments and the rect where the renderer
    # pasted its glyph canvas.
    if info.get('id') is not None:
        bubble['id'] = info['id']
    for k in ('furi', 'line_ids', 'raw_tr'):
        if info.get(k):
            bubble[k] = info[k]
    if info.get('tbox'):
        bubble['tbox'] = _study_norm(*info['tbox'], W, H)
    # The area the renderer laid the text into, as a polygon of page fractions. Texts sharing a
    # balloon carry the same one.
    if info.get('shape'):
        bubble['shape'] = [[round(x / W, 4), round(y / H, 4)] for x, y in info['shape']]
    return bubble


# ── furigana (ruby) segmentation for study-mode source text ───────────────────────────────
# pykakasi splits Japanese text into words with kana readings; each source line becomes a list of
# [text, reading|None] segments where a reading covers only the kanji run (shared kana at the
# word's edges — okurigana/prefixes — are trimmed out of the ruby). The reader decides whether
# to SHOW furigana (it gates on the gallery's language), so this only skips work when the text
# plainly has no kanji or pykakasi isn't installed.

_KAKASI = None


def _has_kanji(s):
    # CJK unified ideographs (incl. ext. A) + compatibility ideographs + iteration marks.
    return any('㐀' <= c <= '鿿' or '豈' <= c <= '﫿' or c in '々〆' for c in s)


def _furi_seg_append(segs, text, ruby):
    """Append a segment, merging consecutive no-ruby runs to keep the payload compact."""
    if ruby is None and segs and segs[-1][1] is None:
        segs[-1][0] += text
    else:
        segs.append([text, ruby])


def _furi_lines(lines):
    """Per-line ruby segments for `lines`, or None when nothing gets a reading."""
    global _KAKASI
    if not any(_has_kanji(line) for line in lines):
        return None
    if _KAKASI is None:
        try:
            import pykakasi
            _KAKASI = pykakasi.kakasi()
        except Exception:
            _KAKASI = False
    if _KAKASI is False:
        return None
    out = []
    any_ruby = False
    for line in lines:
        segs = []
        try:
            items = _KAKASI.convert(line)
        except Exception:
            items = []
        if not items:
            segs.append([line, None])
            out.append(segs)
            continue
        for item in items:
            orig = item.get('orig') or ''
            hira = item.get('hira') or ''
            if not orig:
                continue
            if not hira or not _has_kanji(orig):
                _furi_seg_append(segs, orig, None)
                continue
            # Trim kana the word shares with its reading at both ends so the ruby sits over
            # the kanji only (お預け/おあずけ → お + 預け⟨あず⟩… etc.).
            p = 0
            while p < len(orig) and p < len(hira) and orig[p] == hira[p]:
                p += 1
            s = 0
            while s < len(orig) - p and s < len(hira) - p and orig[len(orig) - 1 - s] == hira[len(hira) - 1 - s]:
                s += 1
            core_o, core_h = orig[p:len(orig) - s], hira[p:len(hira) - s]
            if not core_o or not core_h:
                _furi_seg_append(segs, orig, None)
                continue
            if p:
                _furi_seg_append(segs, orig[:p], None)
            _furi_seg_append(segs, core_o, core_h)
            any_ruby = True
            if s:
                _furi_seg_append(segs, orig[len(orig) - s:], None)
        out.append(segs)
    return out if any_ruby else None


def _study_img_data_url(arr, mode_, fmt='PNG', **save_kw):
    import io as _io
    import base64
    buf = _io.BytesIO()
    Image.fromarray(arr, mode_).save(buf, format=fmt, **save_kw)
    mime = 'image/webp' if fmt == 'WEBP' else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


# Given to page-result hooks in place of the image when a page's study data stands in for it.
STUDY_LAYERS = 'study-layers'


def _study_bg(inpainted, quality):
    """The shared inpaint bg: opaque art (no text), so WebP is far smaller than PNG at quality
    the eye can't tell apart."""
    return _study_img_data_url(inpainted, 'RGB', 'WEBP', quality=quality, method=6)


def _build_page_layers_job(rendered, inpainted, infos, W, H, standalone=False):
    """CPU-bound study layers for one page — the process-pool job. EXACTNESS INVARIANT
    (unchanged): every pixel where the final render differs from the inpaint is assigned to
    exactly ONE bubble, whose layer stores the final render's RGB at full alpha, so reassembling
    all bubble layers over the inpaint reproduces the final page pixel-for-pixel. Ownership is
    per pixel: the render box that contains it (nearest box center on overlap), else the nearest
    render box — so adjacent bubbles split cleanly at their boundary.

    `standalone`: the render IS the page's output (no transparency, nothing resized after
    rendering). Then, unless the fallback below approximated the partition, the layers rebuild the
    page exactly and it is sent as them alone (`rebuilds`), with the bg at the page's own quality."""
    diff = np.abs(rendered.astype(np.int16) - inpainted.astype(np.int16)).max(axis=2)
    changed = diff > 0
    # Defensive: a renderer that perturbs untouched pixels (full-page roundtrip) would mark
    # everything changed; fall back to the visible-change threshold in that case.
    exact = changed.mean() <= 0.35
    if not exact:
        changed = diff > 8
    ys, xs = np.nonzero(changed)
    if xs.size == 0:
        return None

    xf = xs.astype(np.float32)
    yf = ys.astype(np.float32)
    best_d = np.full(xs.shape, np.inf, dtype=np.float32)
    owner = np.zeros(xs.shape, dtype=np.int32)
    for ri, info in enumerate(infos):
        bx1, by1, bx2, by2 = info['rbox']
        # squared rect-distance to the render box (0 inside) + a tiny center-distance term that
        # deterministically breaks ties between overlapping boxes.
        dx = np.maximum(np.maximum(bx1 - xf, 0.0), xf - (bx2 - 1))
        dy = np.maximum(np.maximum(by1 - yf, 0.0), yf - (by2 - 1))
        d = dx * dx + dy * dy
        cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        d += ((xf - cx) ** 2 + (yf - cy) ** 2) * np.float32(1e-7)
        m = d < best_d
        best_d[m] = d[m]
        owner[m] = ri

    bubbles = []
    for ri, info in enumerate(infos):
        sel = owner == ri
        if not sel.any():
            continue
        rx, ry = xs[sel], ys[sel]
        gx1, gy1 = int(rx.min()), int(ry.min())
        gx2, gy2 = int(rx.max()) + 1, int(ry.max()) + 1
        # Full-page transparent layer holding exactly this bubble's pixels: RGB from the final
        # render (antialiasing against the inpaint already baked in), alpha 255 so compositing
        # over the inpaint bg reproduces the render exactly.
        text_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        text_rgba[ry, rx, :3] = rendered[ry, rx]
        text_rgba[ry, rx, 3] = 255
        # bg clip region = detection box unioned with the full glyph extent.
        dx1, dy1, dx2, dy2 = info['det']
        bubble = _study_meta_bubble(info, (min(dx1, gx1), min(dy1, gy1), max(dx2, gx2), max(dy2, gy2)), W, H)
        # Lossless WebP keeps the invariant at about a quarter of PNG's size (a whole page of
        # transparency per bubble); PNG only for a page too long for WebP.
        if max(W, H) <= 16383:
            bubble['text'] = _study_img_data_url(text_rgba, 'RGBA', 'WEBP', lossless=True, quality=100, method=1, exact=True)
        else:
            bubble['text'] = _study_img_data_url(text_rgba, 'RGBA', 'PNG')
        bubbles.append(bubble)
    if not bubbles:
        return None
    rebuilds = standalone and exact
    # Standing in for the translated page, the bg keeps the page's own quality (q95); otherwise
    # it only shows behind a revealed bubble.
    out = {'page': {'w': W, 'h': H}, 'bg': _study_bg(inpainted, 95 if rebuilds else 90), 'bubbles': bubbles}
    if rebuilds:
        out['rebuilds'] = True
    return out
