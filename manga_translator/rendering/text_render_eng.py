import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

from .text_render import get_char_glyph, put_char_horizontal, add_color
from .ballon_extractor import extract_ballon_region
from .bubble_seg import assign_regions, detect_bubbles
from ..utils import TextBlock, rect_distance

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PUNSET_RIGHT_ENG = {'.', '?', '!', ':', ';', ')', '}', "\""}


class Textline:
    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0, spacing: int = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)
        self.num_words = 0
        if text:
            self.num_words += 1
        self.spacing = 0
        self.add_spacing(spacing)

    def append_right(self, word: str, w_len: int, delimiter: str = ''):
        self.text = self.text + delimiter + word
        if word:
            self.num_words += 1
        self.length += w_len

    def append_left(self, word: str, w_len: int, delimiter: str = ''):
        self.text = word + delimiter + self.text
        if word:
            self.num_words += 1
        self.length += w_len

    def add_spacing(self, spacing: int):
        self.spacing = spacing
        self.pos_x -= spacing
        self.length += 2 * spacing

    def strip_spacing(self):
        self.length -= self.spacing * 2
        self.pos_x += self.spacing
        self.spacing = 0

def render_lines(
    textlines: List[Textline],
    canvas_h: int,
    canvas_w: int,
    font_size: int,
    stroke_width: int,
    line_spacing: int = 0.01,
    fg: Tuple[int] = (0, 0, 0),
    bg: Tuple[int] = (255, 255, 255)) -> Image.Image:

    # bg_size = int(max(font_size * 0.1, 1)) if bg is not None else 0
    bg_size = stroke_width
    spacing_y = int(font_size * (line_spacing or 0.01))

    # make large canvas
    canvas_w = max([l.length for l in textlines]) + (font_size + bg_size) * 2
    canvas_h = font_size * len(textlines) + spacing_y * (len(textlines) - 1)  + (font_size + bg_size) * 2
    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    # pen (x, y)
    pen_orig = [font_size + bg_size, font_size + bg_size]

    # write stuff
    for line in textlines:
        pen_line = pen_orig.copy()
        pen_line[0] += line.pos_x # center
        for c in line.text:
            offset_x = put_char_horizontal(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
            pen_line[0] += offset_x
        pen_orig[1] += spacing_y + font_size

    # colorize
    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)

    # rect
    x, y, width, height = cv2.boundingRect(canvas_border)
    return Image.fromarray(line_box[y:y+height, x:x+width])

    # c = Image.new('RGBA', (canvas_w, canvas_h), color = (0, 0, 0, 0))
    # d = ImageDraw.Draw(c)
    # d.fontmode = 'L'
    # for line in lines:
    #     d.text((line.pos_x, line.pos_y), line.text, font=font, fill=font_color, stroke_width=font_size, stroke_fill=stroke_color)
    # return c

def seg_eng(text: str) -> List[str]:
    """
    Extracts every word from text parameter
    """
    # TODO: replace with regexes

    text = text.strip().upper().replace('  ', ' ').replace(' .', '.').replace('\n', ' ')
    processed_text = ''

    # dumb way to ensure spaces between words
    text_len = len(text)
    for ii, c in enumerate(text):
        if c in PUNSET_RIGHT_ENG and ii < text_len - 1:
            next_c = text[ii + 1]
            if next_c.isalpha() or next_c.isnumeric():
                processed_text += c + ' '
            else:
                processed_text += c
        else:
            processed_text += c

    word_list = processed_text.split(' ')
    word_num = len(word_list)
    if word_num <= 1:
        return word_list

    words = []
    skip_next = False
    for ii, word in enumerate(word_list):
        if skip_next:
            skip_next = False
            continue
        if len(word) < 3:
            append_left, append_right = False, False
            len_word, len_next, len_prev = len(word), -1, -1
            if ii < word_num - 1:
                len_next = len(word_list[ii + 1])
            if ii > 0:
                len_prev = len(words[-1])
            cond_next = (len_word == 2 and len_next <= 4) or len_word == 1
            cond_prev = (len_word == 2 and len_prev <= 4) or len_word == 1
            if len_next > 0 and len_prev > 0:
                if len_next < len_prev:
                    append_right = cond_next
                else:
                    append_left = cond_prev
            elif len_next > 0:
                append_right = cond_next
            elif len_prev:
                append_left = cond_prev

            if append_left:
                words[-1] = words[-1] + ' ' + word
            elif append_right:
                words.append(word + ' ' + word_list[ii + 1])
                skip_next = True
            else:
                words.append(word)
            continue
        words.append(word)
    return words

def layout_lines_aligncenter(
    mask: np.ndarray, 
    words: List[str], 
    word_lengths: List[int], 
    delimiter_len: int, 
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    max_central_width: float = np.inf,
    word_break: bool = False)->List[Textline]:

    m = cv2.moments(mask)
    mask = 255 - mask
    centroid_y = int(m['m01'] / m['m00'])
    centroid_x = int(m['m10'] / m['m00'])

    # layout the central line, the center word is approximately aligned with the centroid of the mask
    num_words = len(words)
    len_left, len_right = [], []
    wlst_left, wlst_right = [], []
    sum_left, sum_right = 0, 0
    if num_words > 1:
        wl_array = np.array(word_lengths, dtype=np.float64)
        wl_cumsums = np.cumsum(wl_array)
        wl_cumsums = wl_cumsums - wl_cumsums[-1] / 2 - wl_array / 2
        central_index = np.argmin(np.abs(wl_cumsums))

        if central_index > 0:
            wlst_left = words[:central_index]
            len_left = word_lengths[:central_index]
            sum_left = np.sum(len_left)
        if central_index < num_words - 1:
            wlst_right = words[central_index + 1:]
            len_right = word_lengths[central_index + 1:]
            sum_right = np.sum(len_right)
    else:
        central_index = 0

    pos_y = centroid_y - line_height // 2
    pos_x = centroid_x - word_lengths[central_index] // 2

    bh, bw = mask.shape[:2]
    central_line = Textline(words[central_index], pos_x, pos_y, word_lengths[central_index], spacing)
    line_bottom = pos_y + line_height
    while sum_left > 0 or sum_right > 0:
        left_valid, right_valid = False, False

        if sum_left > 0:
            new_len_l = central_line.length + len_left[-1] + delimiter_len
            new_x_l = centroid_x - new_len_l // 2
            new_r_l = new_x_l + new_len_l
            if (new_x_l > 0 and new_r_l < bw):
                if mask[pos_y: line_bottom, new_x_l].sum()==0 and mask[pos_y: line_bottom, new_r_l].sum() == 0:
                    left_valid = True
        if sum_right > 0:
            new_len_r = central_line.length + len_right[0] + delimiter_len
            new_x_r = centroid_x - new_len_r // 2
            new_r_r = new_x_r + new_len_r
            if (new_x_r > 0 and new_r_r < bw):
                if mask[pos_y: line_bottom, new_x_r].sum()==0 and mask[pos_y: line_bottom, new_r_r].sum() == 0:
                    right_valid = True

        insert_left = False
        if left_valid and right_valid:
            if sum_left > sum_right:
                insert_left = True
        elif left_valid:
            insert_left = True
        elif not right_valid:
            break

        if insert_left:
            central_line.append_left(wlst_left.pop(-1), len_left[-1] + delimiter_len, delimiter)
            sum_left -= len_left.pop(-1)
            central_line.pos_x = new_x_l
        else:
            central_line.append_right(wlst_right.pop(0), len_right[0] + delimiter_len, delimiter)
            sum_right -= len_right.pop(0)
            central_line.pos_x = new_x_r
        if central_line.length > max_central_width:
            break

    central_line.strip_spacing()
    lines = [central_line]

    # layout bottom half
    if sum_right > 0:
        w, wl = wlst_right.pop(0), len_right.pop(0)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y + line_height // 2
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.append(line)
        sum_right -= wl
        while sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
                mask[pos_y: line_bottom, right_x].sum() > 0:
                line_valid = False
            else:
                line_valid = True
            if line_valid:
                line.append_right(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line_valid = False
                    if sum_right > 0:
                        w, wl = wlst_right.pop(0), len_right.pop(0)
                        sum_right -= wl
                    else:
                        line.strip_spacing()
                        break

            if not line_valid:
                pos_x = centroid_x - wl // 2
                pos_y = line_bottom
                line_bottom += line_height
                line.strip_spacing()
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.append(line)

    # layout top half
    if sum_left > 0:
        w, wl = wlst_left.pop(-1), len_left.pop(-1)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y - line_height // 2 - line_height
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.insert(0, line)
        sum_left -= wl
        while sum_left > 0:
            w, wl = wlst_left.pop(-1), len_left.pop(-1)
            sum_left -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
                mask[pos_y: line_bottom, right_x].sum() > 0:
                line_valid = False
            else:
                line_valid = True
            if line_valid:
                line.append_left(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line_valid = False
                    if sum_left > 0:
                        w, wl = wlst_left.pop(-1), len_left.pop(-1)
                        sum_left -= wl
                    else:
                        line.strip_spacing()
                        break

            if not line_valid:
                pos_x = centroid_x - wl // 2
                pos_y -= line_height
                line_bottom = pos_y + line_height
                line.strip_spacing()
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.insert(0, line)

    # rbgmsk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.circle(rbgmsk, (centroid_x, centroid_y), 10, (255, 0, 0))
    # for line in lines:
    #     cv2.rectangle(rbgmsk, (line.pos_x, line.pos_y), (line.pos_x + line.length, line.pos_y + line_height), (0, 255, 0))
    # cv2.imshow('mask', rbgmsk)
    # cv2.waitKey(0)

    return lines


def _glyph_band(font_size: int) -> Tuple[int, int]:
    """(glyph_h, inset): cap-to-baseline footprint the glyphs occupy within the em, and its
    top inset. Shared by the fit (row measurement) and the paste (block extents) so what is
    measured matches where the ink lands."""
    glyph_h = max(1, int(round(0.72 * font_size)))
    return glyph_h, (font_size - glyph_h) // 2


def prep_scanline(mask: np.ndarray, pad: int):
    """Precompute the per-row centred inside-width of a balloon, once per region.

    `mask` is 255 inside. `pad` erodes the balloon first (this is the padding knob). Returns
    (centred_widths[h], cx, cy): for each row, the width of the inside run through cx (0 where cx
    is outside the balloon at that row), and the balloon centroid. Reused across every font-size
    probe so the search itself touches no pixels.
    """
    inside = (mask > 0).astype(np.uint8)
    if pad > 0:
        # The mask is cropped to the balloon's bounding box, so a flat balloon edge lies ON the
        # crop border; erode treats out-of-image as inside, which left flat tops/sides with zero
        # padding. Pad the crop with background first so every edge erodes.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        bordered = cv2.copyMakeBorder(inside, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        eroded = cv2.erode(bordered, k)[pad:-pad, pad:-pad]
        if eroded.any():
            inside = eroded

    m = cv2.moments(inside, binaryImage=True)
    h, w = inside.shape
    if m['m00'] == 0:
        return np.zeros(h, dtype=np.int32), w // 2, h // 2
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    cx = min(max(cx, 0), w - 1)

    # Per row, the width of the inside run through cx (0 where cx is outside the balloon), computed
    # vectorised: the nearest outside pixel left and right of cx bounds the centred run.
    outside = inside == 0
    left_cols = np.where(outside[:, :cx + 1], np.arange(cx + 1), -1).max(axis=1) + 1
    right_cols = np.where(outside[:, cx:], np.arange(cx, w), w).min(axis=1) - 1
    widths = 2 * np.minimum(cx - left_cols, right_cols - cx)
    widths[~inside[:, cx].astype(bool)] = 0
    return np.maximum(widths, 0).astype(np.int32), cx, cy


def _greedy_wrap(word_lengths, delimiter_len, bands):
    """Greedily pack words into successive line bands of the given pixel widths.

    Returns a list of (start, end) word-index ranges if every word is placed within len(bands)
    lines, else None (a band too narrow even for one word, or words left over)."""
    n = len(word_lengths)
    i = 0
    lines = []
    for band_w in bands:
        if band_w <= 0:
            return None
        start = i
        cur = 0
        while i < n:
            add = word_lengths[i] + (delimiter_len if cur > 0 else 0)
            if cur + add <= band_w:
                cur += add
                i += 1
            else:
                break
        if i == start:
            return None  # widest word does not fit this band
        lines.append((start, i))
        if i >= n:
            return lines  # all words placed (may use fewer than len(bands) lines)
    return None  # ran out of lines with words remaining


# Demerit weights for _balanced_wrap, all on the scale of the squared-relative-slack term
# (typical line slack lands around 0.01-0.2). Knuth-Plass-style: the DP minimises their sum.
_WRAP_SMOOTH = 0.35    # squared difference between consecutive lines' fill ratios
_WRAP_ORPHAN = 0.18    # a lone word filling less than half its line's band
_WRAP_BAD_BREAK = 0.10   # breaking right after an article/preposition/etc.
_WRAP_GOOD_BREAK = -0.05  # breaking after sentence/clause punctuation

# Words a letterer keeps attached to what follows: breaking a line right after one of these
# reads badly ("...THE / ROAR"). Matched case-insensitively on the word sans punctuation.
_GLUE_WORDS = frozenset('''
    a an the of to in on at by for with from into over under and or but nor so if as than
    my your his her its our their is are was were be been am will would can could shall
    should must may might
'''.split())

_LINE_END_PUNCT = '.!?,:;…'


def _break_penalties(words):
    """Per-gap demerit for breaking after words[i] (length len(words)-1): negative after
    clause punctuation (a natural pause), positive after a glue word, a bare numeral
    ("GIVE ME 5 / DAYS"), or a lone punctuation token (an opening quote binds to what
    follows). A token may be a seg_eng chunk of several words — what matters for the break
    is its last word."""
    pens = []
    for word in words[:-1]:
        bare = word.rsplit(' ', 1)[-1].rstrip('"”’\')')
        core = bare.strip('".,!?…“”:;')
        if bare and bare[-1] in _LINE_END_PUNCT:
            pens.append(_WRAP_GOOD_BREAK)
        elif core.lower() in _GLUE_WORDS or core.isdigit() or not core:
            pens.append(_WRAP_BAD_BREAK)
        else:
            pens.append(0.0)
    return pens


def _balanced_wrap(word_lengths, delimiter_len, bands, gap_pens=None):
    """Split words into exactly len(bands) consecutive, non-empty lines, line i no wider than
    bands[i], minimising Knuth-Plass-style demerits: squared relative slack (each line's
    length tracks the width available at its rows, so the block takes the balloon's shape),
    a smoothness term between consecutive lines' fill ratios (a short line costs once at the
    block's tips but twice in its middle, which pushes short lines outward and keeps the
    silhouette convex like a letterer's diamond), an orphan term for a lone short word on a
    line, and per-gap break penalties from `_break_penalties`.

    The smoothness term needs the previous line's fill in the DP state; fills are bucketed
    into a few classes to keep that tractable. Returns (splits, mean demerits) where splits
    is a (start, end) word range per line, or None if no such split exists."""
    n = len(word_lengths)
    k = len(bands)
    if k > n or any(b <= 0 for b in bands):
        return None
    prefix = [0]
    for w in word_lengths:
        prefix.append(prefix[-1] + w)

    B = 8
    centers = [(2 * b + 1) / (2 * B) for b in range(B)]
    INF = float('inf')
    # cost[j][e][b]: best demerits laying words 0..e-1 on lines 1..j, line j's fill in bucket b
    cost = [[[INF] * B for _ in range(n + 1)] for _ in range(k + 1)]
    parent = [[[None] * B for _ in range(n + 1)] for _ in range(k + 1)]
    cost[0][0][0] = 0.0
    for j in range(1, k + 1):
        band = bands[j - 1]
        prev = cost[j - 1]
        cur = cost[j]
        par = parent[j]
        # line j takes words s..e-1 (at least one), leaving enough words for the lines after it
        for e in range(j, n - (k - j) + 1):
            for s in range(j - 1, e):
                w = prefix[e] - prefix[s] + delimiter_len * (e - s - 1)
                if w > band:
                    continue
                f = w / band
                line_cost = (1.0 - f) ** 2
                if e - s == 1 and f < 0.5:
                    line_cost += _WRAP_ORPHAN
                if j == 1:
                    base = prev[0][0]
                    if base >= INF:
                        continue
                    total, bp = base + line_cost, 0
                else:
                    if gap_pens is not None:
                        line_cost += gap_pens[s - 1]
                    total, bp = INF, -1
                    for b0 in range(B):
                        pc = prev[s][b0]
                        if pc >= INF:
                            continue
                        c = pc + line_cost + _WRAP_SMOOTH * (centers[b0] - f) ** 2
                        if c < total:
                            total, bp = c, b0
                    if bp < 0:
                        continue
                b = min(B - 1, int(f * B))
                if total < cur[e][b]:
                    cur[e][b] = total
                    par[e][b] = (s, bp)
    end_b = min(range(B), key=lambda b: cost[k][n][b])
    if cost[k][n][end_b] >= INF:
        return None
    total = cost[k][n][end_b]
    splits = []
    e, b = n, end_b
    for j in range(k, 0, -1):
        s, b = parent[j][e][b]
        splits.append((s, e))
        e = s
    splits.reverse()
    return splits, total / k


def scanline_polish(centred_widths, cx, cy, bh, words, word_lengths, delimiter_len,
                    font_size, spacing_y, sw, k_hint, delimiter=' '):
    """Re-break already-fitted text so the lines fill the balloon's geometry.

    Runs once, at the font size the fit search settled on: tries line counts at and just above
    the greedy solution's, splitting words with _balanced_wrap. The block stays centred on the
    balloon's rows — the fit only ever accepts centred layouts, so a centred re-break is always
    feasible at k_hint (the greedy split is a witness) and drifting for a better fill would
    visibly misalign the text. Returns centred Textlines, or None to keep the greedy layout
    (single word or oversized text — the greedy result is always still valid)."""
    n = len(words)
    if n < 2 or n > 60:
        return None
    pitch = font_size + spacing_y
    glyph_h, inset = _glyph_band(font_size)
    max_lines = max(1, int(bh // max(1, pitch)) + 1)
    gap_pens = _break_penalties(words)
    best_score, best_lines = None, None
    for k in range(k_hint, min(max_lines, k_hint + 2) + 1):
        if k > n:
            break
        block_h = (k - 1) * pitch + font_size
        top = int(round(cy - block_h / 2.0))
        if top < 0 or top + block_h > bh:
            continue
        bands = []
        for i in range(k):
            gy = top + i * pitch + inset
            bands.append(int(centred_widths[gy:gy + glyph_h].min()) - 2 * sw)
        got = _balanced_wrap(word_lengths, delimiter_len, bands, gap_pens)
        if got is None:
            continue
        splits, mean_slack = got
        if best_score is None or mean_slack < best_score:
            out = []
            for row, (s, e) in enumerate(splits):
                length = sum(word_lengths[s:e]) + delimiter_len * (e - s - 1)
                out.append(Textline(delimiter.join(words[s:e]),
                                    int(cx - length // 2), int(top + row * pitch), length))
            best_score, best_lines = mean_slack, out
    return best_lines


def scanline_fit(centred_widths, cx, cy, bh, words, word_lengths, delimiter_len,
                 font_size, spacing_y, sw, delimiter=' '):
    """Lay text into the balloon following its contour, at one font size.

    Places a K-line block centred on the balloon, measuring each line's usable width from the
    actual inside span at the rows that line's glyphs will occupy — so lines follow the balloon's
    width profile and the block is bounded to the balloon vertically. Uses the *render* pitch
    (`font_size + spacing_y`) and glyph height (`font_size`) so what is measured matches what is
    drawn. Returns centred Textlines, or None if the text cannot fit at this size.
    """
    pitch = font_size + spacing_y
    glyph_h, inset = _glyph_band(font_size)
    max_lines = max(1, int(bh // max(1, pitch)) + 1)
    for k in range(1, max_lines + 1):
        block_h = (k - 1) * pitch + font_size
        top = int(round(cy - block_h / 2.0))
        if top < 0 or top + block_h > bh:
            return None  # block cannot sit inside the balloon vertically at this size
        bands = []
        for i in range(k):
            gy = top + i * pitch + inset  # measure width over the glyph rows, not the full em
            bands.append(int(centred_widths[gy:gy + glyph_h].min()) - 2 * sw)
        placed = _greedy_wrap(word_lengths, delimiter_len, bands)
        # Only accept a layout that uses all k lines: greedy finishing early would leave the
        # block sized (and centred) for k lines but drawn with fewer, riding high in the
        # balloon. Rejecting it lets the font search settle a notch smaller and stay centred.
        if placed is not None and len(placed) == k:
            out = []
            for row, (s, e) in enumerate(placed):
                length = sum(word_lengths[s:e]) + delimiter_len * (e - s - 1)
                out.append(Textline(delimiter.join(words[s:e]),
                                    int(cx - length // 2), int(top + row * pitch), length))
            return out
    return None

def render_textblock_list_eng(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_color = (0, 0, 0),
    stroke_color = (255, 255, 255),
    delimiter: str = ' ',
    line_spacing: int = 0.01,
    stroke_width: float = 0.1,
    size_tol: float = 1.0,
    ballonarea_thresh: float = 2,
    downscale_constraint: float = 0.7,
    original_img: np.ndarray = None,
    disable_font_border: bool = False,
    verbose: bool = False,
    bubble_padding_ratio: float = 0.06,
    bubble_fill_upscale: float = 1.4,
    page_bubbles: List[np.ndarray] = None
) -> np.ndarray:

    r"""
    Args:
        downscale_constraint (float, optional): minimum scaling down ratio, prevent rendered text from being too small
        ref_textballon (bool, optional): take text balloons as reference for text layout. 
        original_img (np.ndarray, optional): original image used to extract text balloons.
    """

    def calculate_font_values(font_size: int, words: List[str]):
        font_size = int(font_size)
        sw = int(font_size * stroke_width)
        line_height = int(font_size * 0.8)
        delimiter_glyph = get_char_glyph(delimiter, font_size, 0)
        delimiter_len = delimiter_glyph.advance.x >> 6
        base_length = -1
        word_lengths = []
        for word in words:
            word_length = 0
            for cdpt in word:
                glyph = get_char_glyph(cdpt, font_size, 0)
                char_offset_x = glyph.metrics.horiAdvance >> 6
                word_length += char_offset_x
            word_lengths.append(word_length)
            if word_length > base_length:
                base_length = word_length
        return font_size, sw, line_height, delimiter_len, base_length, word_lengths

    def inside_ratio(mask: np.ndarray, textlines: List['Textline'], sw: int, line_h: int, y_off: int = 0) -> float:
        """Fraction of a laid-out block's area that lands inside the balloon.

        The total is taken from the line geometry rather than the rasterised map, so lines that
        fall outside the crop entirely still count against it. `y_off`/`line_h` let the caller
        measure the actual glyph band of a scanline fit rather than the 0.8-em line box. The
        rectangle corners are end-inclusive in cv2, hence the -1s, so the map matches `total`.
        """
        total = 0
        lines_map = np.zeros_like(mask, dtype=np.uint8)
        for line in textlines:
            total += (line.length + 2 * sw) * line_h
            cv2.rectangle(lines_map, (line.pos_x - sw, line.pos_y + y_off),
                          (line.pos_x + line.length + sw - 1, line.pos_y + y_off + line_h - 1), 255, -1)
        if total <= 0:
            return 1.0
        return min(1.0, int(np.count_nonzero(cv2.bitwise_and(lines_map, mask))) / total)

    img_pil = Image.fromarray(img)


    # Initialize enlarge ratios
    for region in text_regions:
        region.enlarge_ratio = 1
        region.enlarged_xyxy = region.xyxy.copy()

    def update_enlarged_xyxy(region):
        region.enlarged_xyxy = region.xyxy.copy()
        w_diff, h_diff = ((region.xywh[2:] * region.enlarge_ratio) - region.xywh[2:].astype(np.float64)) // 2
        region.enlarged_xyxy[0] -= w_diff
        region.enlarged_xyxy[2] += w_diff
        region.enlarged_xyxy[1] -= h_diff
        region.enlarged_xyxy[3] += h_diff

    # Adjust enlarge ratios relative to each other to reduce intersections
    for region in text_regions:
        # If it wasn't changed below already
        if region.enlarge_ratio == 1:
            # The larger the aspect ratio the more it should try to enlarge the bubble
            region.enlarge_ratio = min(max(region.xywh[2] / region.xywh[3], region.xywh[3] / region.xywh[2]) * 1.5, 3)
            update_enlarged_xyxy(region)

        for region2 in text_regions:
            if region is region2:
                continue

            if rect_distance(*region.enlarged_xyxy, *region2.enlarged_xyxy) == 0: # if intersect
                # Get prior distance and adjust both enlargement ratios accordingly
                d = rect_distance(*region.xyxy, *region2.xyxy)
                l1 = (region.xywh[2] + region.xywh[3]) / 2
                l2 = (region2.xywh[2] + region2.xywh[3]) / 2
                region.enlarge_ratio = d / (2 * l1) + 1
                region2.enlarge_ratio = d / (2 * l2) + 1
                update_enlarged_xyxy(region)
                update_enlarged_xyxy(region2)
                # print('Reducing enlarge ratio to prevent intersection')
                # print(region.translation, region.enlarged_xyxy, region.enlarge_ratio)
                # print('>->', region2.translation, region2.enlarged_xyxy, region2.enlarge_ratio)

    # Segmented balloons are trusted enough to scale text against (the overflow shrink); the
    # contour extractor below is not, so its regions skip that correction. Both mask kinds
    # feed the contour-following fit.
    segmented = None
    if original_img is not None:
        if page_bubbles is None:
            page_bubbles = detect_bubbles(original_img)
        if page_bubbles:
            segmented = assign_regions(page_bubbles, [r.xyxy for r in text_regions])

    for idx, region in enumerate(text_regions):
        words = seg_eng(region.translation)
        if not words:
            continue

        font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(region.font_size, words)

        # Extract ballon region
        bubble = segmented[idx] if segmented is not None else None
        trusted_mask = bubble is not None
        if trusted_mask:
            ballon_mask, xyxy = bubble
        else:
            # non-dl textballon segmentation
            ballon_mask, xyxy = extract_ballon_region(original_img, region.xywh, enlarge_ratio=region.enlarge_ratio)
        ballon_area = (ballon_mask > 0).sum()
        rotated, rx, ry = False, 0, 0

        if verbose:
            # Capture the balloon the fit will use, in page coords, before the rotation block
            # below mutates ballon_mask. This is what bubbles.png draws per region.
            region._bubble_source = 'segmented' if trusted_mask else 'contour'
            region._bubble_poly = None
            region._bubble_poly_pad = None
            cnts, _ = cv2.findContours((ballon_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                poly = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
                poly[:, 0] += int(xyxy[0])
                poly[:, 1] += int(xyxy[1])
                region._bubble_poly = poly
            # The padded boundary the wrap actually keeps text inside (same erosion as
            # prep_scanline), so bubbles.png shows the margin, not just the balloon.
            pad_dbg = int(round(bubble_padding_ratio * min(ballon_mask.shape)))
            if pad_dbg > 0:
                k_dbg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad_dbg + 1, 2 * pad_dbg + 1))
                bordered_dbg = cv2.copyMakeBorder((ballon_mask > 0).astype(np.uint8), pad_dbg, pad_dbg,
                                                  pad_dbg, pad_dbg, cv2.BORDER_CONSTANT, value=0)
                eroded_dbg = cv2.erode(bordered_dbg, k_dbg)[pad_dbg:-pad_dbg, pad_dbg:-pad_dbg]
                cnts, _ = cv2.findContours(eroded_dbg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    poly = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
                    poly[:, 0] += int(xyxy[0])
                    poly[:, 1] += int(xyxy[1])
                    region._bubble_poly_pad = poly

        if abs(region.angle) > 3:
            rotated = True
            region_angle_rad = np.deg2rad(region.angle)
            region_angle_sin = np.sin(region_angle_rad)
            region_angle_cos = np.cos(region_angle_rad)
            rotated_ballon_mask = Image.fromarray(ballon_mask).rotate(region.angle, expand=True)
            rotated_ballon_mask = np.array(rotated_ballon_mask)

            region.angle %= 360
            if region.angle > 0 and region.angle <= 90:
                ry = abs(ballon_mask.shape[1] * region_angle_sin)
            elif region.angle > 90 and region.angle <= 180:
                rx = abs(ballon_mask.shape[1] * region_angle_cos)
                ry = rotated_ballon_mask.shape[0]
            elif region.angle > 180 and region.angle <= 270:
                ry = abs(ballon_mask.shape[0] * region_angle_cos)
                rx = rotated_ballon_mask.shape[1]
            else:
                rx = abs(ballon_mask.shape[0] * region_angle_sin)
            ballon_mask = rotated_ballon_mask

        line_width = sum(word_lengths) + delimiter_len * (len(word_lengths) - 1)
        region_area = line_width * line_height + delimiter_len * (len(words) - 1) * line_height
        area_ratio = ballon_area / region_area
        resize_ratio = 1

        # In many cases this code makes the font size too small.
        # # if ballon_area is smaller than 2*region_area
        # if area_ratio < ballonarea_thresh:
        #     # resize so that it is 2*region_area
        #     resize_ratio = ballonarea_thresh / area_ratio
        #     ballon_area = int(resize_ratio * ballon_area) # = ballonarea_thresh * line_area
        #     resize_ratio = min(np.sqrt(resize_ratio), (1/downscale_constraint)**2)
        #     rx *= resize_ratio
        #     ry *= resize_ratio
        #     ballon_mask = cv2.resize(ballon_mask, (int(resize_ratio * ballon_mask.shape[1]), int(resize_ratio * ballon_mask.shape[0])))

        # new region bbox
        region_x, region_y, region_w, region_h = cv2.boundingRect(cv2.findNonZero(ballon_mask))

        base_length_word = words[max(enumerate(word_lengths), key = lambda x: x[1])[0]]
        if len(base_length_word) == 0 :
            continue

        textlines = None
        fit_floored = False
        scan_fitted = False
        # Fill the balloon following its actual contour: for the largest font whose text fits,
        # each line takes the real inside-width at the rows it occupies (so lines follow the
        # balloon's width profile, with `pad` kept clear of its edge) and the block is bounded
        # to the balloon vertically. Runs against segmented and contour masks alike so word
        # wrap always tracks the region shape; `trusted_mask` only governs the overflow shrink
        # further down. A binary search over font size replaces the old estimate-and-shrink,
        # which fit a roughly rectangular block against two endpoints and routinely overflowed.
        pad = int(round(bubble_padding_ratio * min(ballon_mask.shape)))
        centred_widths, cx_c, cy_c = prep_scanline(ballon_mask, pad)
        bh = ballon_mask.shape[0]
        lo = max(1, int(font_size * downscale_constraint))
        # A roomy balloon lets the text grow beyond the source font size to fill it, the way a
        # letterer would pick the size for the balloon. Only against a trusted mask: growing
        # into an unreliable contour mask would overflow the real balloon.
        hi = max(lo, int(font_size * (bubble_fill_upscale if trusted_mask else 1.0)))
        # The scanline fit and polish work at true word granularity. seg_eng glues short words
        # into length-based chunks ("IT TO"), which the demerit wrap would otherwise treat as
        # one unbreakable token and which can also block a fit the search could take a notch
        # larger. Splitting them back into words only ever adds break options, so the fitted
        # size stays feasible; the demerit penalties (glue words, punctuation) do the phrase
        # binding the chunks used to. The chunk list survives only for the centred fallback
        # below, which has no demerit machinery and leans on gluing as its orphan guard.
        words_fine = [w for chunk in words for w in chunk.split(' ') if w]
        a, b = lo, hi
        best = None
        while a <= b:
            mid = (a + b) // 2
            vals = calculate_font_values(mid, words_fine)
            spacing_y = int(mid * (line_spacing or 0.01))
            lines = scanline_fit(centred_widths, cx_c, cy_c, bh, words_fine, vals[5], vals[3],
                                 mid, spacing_y, vals[1], delimiter=delimiter)
            if lines is not None:
                best = (vals, lines)
                a = mid + 1  # try larger
            else:
                b = mid - 1  # too big, shrink
        if best is not None:
            (font_size, sw, line_height, delimiter_len, base_length, word_lengths), textlines = best
            scan_fitted = True
            # The greedy first-fit block found by the search is feasible but ragged; re-break
            # the words so each line fills the width its rows actually have in the balloon.
            spacing_y = int(font_size * (line_spacing or 0.01))
            polished = scanline_polish(centred_widths, cx_c, cy_c, bh, words_fine, word_lengths,
                                       delimiter_len, font_size, spacing_y, sw,
                                       len(textlines), delimiter=delimiter)
            if polished is not None:
                textlines = polished
        else:
            # Text can't fit the balloon even at the floor -- fall back to the centred layout
            # (never drops words). For trusted masks lay out at the floor size and let the
            # overflow correction shrink it further; contour regions keep the historical
            # width-heuristic below.
            fit_floored = True
            if trusted_mask:
                font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(lo, words)

        if textlines is None:
            if not trusted_mask:
                lines_needed = len(region.translation) / len(base_length_word)
                lines_available = abs(xyxy[3] - xyxy[1]) // line_height + 1
                font_size_multiplier = max(min(region_w / (base_length + 2*sw), lines_available / lines_needed), downscale_constraint)
                # print(region.translation, font_size, font_size_multiplier, int(font_size * font_size_multiplier))
                if font_size_multiplier < 1:
                    font_size = int(font_size * font_size_multiplier)
                    font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(font_size, words)
            textlines = layout_lines_aligncenter(ballon_mask, words, word_lengths, delimiter_len, line_height, delimiter=delimiter)

        # Record the size actually drawn (post-downscale) so study-mode DOM text can match it.
        region._drawn_font_size = font_size
        # Record the lines exactly as laid out (top to bottom) so study-mode DOM text can
        # reproduce the same line breaks as the drawn image.
        region._drawn_lines = [line.text for line in textlines]

        if verbose:
            # How much of the final block lands inside the balloon, measured in the same frame the
            # layout used (before the pos_x/pos_y shift below), plus whether the fit loop bottomed
            # out. bubbles.png labels each region with these. Scan-fitted blocks are measured over
            # the glyph band the fit actually reserved.
            if scan_fitted:
                glyph_h_dbg, inset_dbg = _glyph_band(font_size)
                region._bubble_fit = inside_ratio(ballon_mask, textlines, sw, glyph_h_dbg, inset_dbg)
            else:
                region._bubble_fit = inside_ratio(ballon_mask, textlines, sw, line_height)
            region._bubble_floored = fit_floored

        if scan_fitted:
            # The fit already centred the block on the balloon's rows; re-centring on the crop
            # bbox would displace it off the rows it was measured against and clip. The paste
            # extents span the glyph band the fit measured (not the historical 0.8-em line box)
            # so the centre-based paste below lands the ink exactly where it was laid out.
            y_offset = 0
        else:
            line_cy = np.array([line.pos_y for line in textlines]).mean() + line_height / 2
            region_cy = region_y + region_h / 2
            y_offset = int(round(np.clip(region_cy - line_cy, -line_height, line_height)))

        lines_x1, lines_x2 = [], []
        for line in textlines:
            lines_x1.append(line.pos_x)
            lines_x2.append(max(line.pos_x, 0) + line.length)
        lines_x1 = np.array(lines_x1)
        lines_x2 = np.array(lines_x2)
        canvas_x1, canvas_x2 = lines_x1.min() - sw, lines_x2.max() + sw
        if scan_fitted:
            glyph_h, band_inset = _glyph_band(font_size)
            canvas_y1 = textlines[0].pos_y + band_inset - sw
            canvas_y2 = textlines[-1].pos_y + band_inset + glyph_h + sw
        else:
            canvas_y1, canvas_y2 = textlines[0].pos_y - sw, textlines[-1].pos_y + line_height + sw
        canvas_h = int(canvas_y2 - canvas_y1)
        canvas_w = int(canvas_x2 - canvas_x1)
        lines_map = np.zeros_like(ballon_mask, dtype=np.uint8)
        for line in textlines:
            # line.pos_y += y_offset
            cv2.rectangle(lines_map, (line.pos_x - sw, line.pos_y + y_offset), (line.pos_x + line.length + sw, line.pos_y + line_height), 255, -1)
            line.pos_x -= canvas_x1
            line.pos_y -= canvas_y1

        region_font_color, region_stroke_color = region.get_font_colors()

        textlines_image = render_lines(textlines, canvas_h, canvas_w, font_size, sw, line_spacing, region_font_color, region_stroke_color)
        rel_cx = ((canvas_x1 + canvas_x2) / 2 - rx) / resize_ratio
        rel_cy = ((canvas_y1 + canvas_y2) / 2 - ry + y_offset) / resize_ratio

        lines_area = np.sum(lines_map)
        lines_area += (max(0, region_y - canvas_y1) + max(0, canvas_y2 - region_h - region_y)) * canvas_w * 255 \
                        + (max(0, region_x - canvas_x1) + max(0, canvas_x2 - region_w - region_x)) * canvas_h * 255

        lines_inside = np.sum(cv2.bitwise_and(lines_map, ballon_mask))
        valid_lines_ratio = lines_area / lines_inside if lines_inside > 0 else 1
        if valid_lines_ratio > 1: # text bbox > ballon area
            resize_ratio = min(resize_ratio * valid_lines_ratio, (1 / downscale_constraint) ** 2)

        if rotated:
            rcx = rel_cx * region_angle_cos - rel_cy * region_angle_sin
            rcy = rel_cx * region_angle_sin + rel_cy * region_angle_cos
            rel_cx = rcx
            rel_cy = rcy
            textlines_image = textlines_image.rotate(-region.angle, expand=True, resample=Image.BILINEAR)
            textlines_image = textlines_image.crop(textlines_image.getbbox())

        abs_cx = rel_cx + xyxy[0]
        abs_cy = rel_cy + xyxy[1]

        # Shrink text that overflowed the balloon. Only safe against a segmented mask: the
        # contour extractor often returns a mask smaller than the real balloon, which reads as
        # overflow and shrinks the text to nothing. The paste below is centre-based, so scaling
        # here keeps the placement computed above.
        if trusted_mask and resize_ratio != 1:
            textlines_image = textlines_image.resize((max(1, int(textlines_image.width / resize_ratio)),
                                                      max(1, int(textlines_image.height / resize_ratio))),
                                                     resample=Image.LANCZOS)
            region._drawn_font_size = font_size / resize_ratio
        abs_x = int(abs_cx - textlines_image.width / 2)
        abs_y = int(abs_cy - textlines_image.height / 2)
        img_pil.paste(textlines_image, (abs_x, abs_y), mask=textlines_image)
        # Record where the glyph canvas actually landed + the drawn line pitch so study-mode
        # DOM text can match the image's placement exactly (the enlarged box is only where
        # layout was ALLOWED, not where it ended up).
        region._drawn_rect = (abs_x, abs_y, abs_x + textlines_image.width, abs_y + textlines_image.height)
        region._drawn_line_height = (font_size + int(font_size * (line_spacing or 0.01))) / font_size
        # cv2.imshow('ballon_region', ballon_region)
        # cv2.imshow('cropped', original_img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
        # cv2.imshow('raw_lines', np.array(raw_lines))
        # cv2.waitKey(0)

    return np.array(img_pil)


def render_bubble_debug(img_rgb: np.ndarray, text_regions: List[TextBlock]) -> np.ndarray:
    """Verbose overlay of the balloon area each eng-rendered region used, and how it was fitted.

    `img_rgb` should be the RENDERED page (the caller passes the renderer's output), so the
    overlay shows the translated text inside its balloon rather than the original text.
    Reads the per-region debug attributes stashed by `render_textblock_list_eng(verbose=True)`.
    Returns a BGR image (ready for `cv2.imwrite`), matching the other verbose stage dumps.

    Legend: green fill/outline = segmented balloon (trusted, overflow shrink armed), orange =
    contour fallback. The thin inner line is the padded boundary the wrap keeps text inside.
    Blue rect = where the text was actually drawn. Per-region label = drawn font size,
    percentage of the block inside the balloon, and FLOOR if the fit loop bottomed out.
    """
    SEG = (0, 200, 0)
    CON = (0, 140, 255)
    RECT = (255, 0, 0)

    canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    debug_regions = [r for r in text_regions if getattr(r, '_bubble_source', None) is not None]

    overlay = canvas.copy()
    for region in debug_regions:
        poly = getattr(region, '_bubble_poly', None)
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(overlay, [poly], SEG if region._bubble_source == 'segmented' else CON)
    # Light tint: the base image now carries the rendered translation, which must stay readable.
    canvas = cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0)

    for region in debug_regions:
        color = SEG if region._bubble_source == 'segmented' else CON
        poly = getattr(region, '_bubble_poly', None)
        if poly is not None and len(poly) >= 3:
            cv2.polylines(canvas, [poly], True, color, 2)
        poly_pad = getattr(region, '_bubble_poly_pad', None)
        if poly_pad is not None and len(poly_pad) >= 3:
            cv2.polylines(canvas, [poly_pad], True, color, 1)

        rect = getattr(region, '_drawn_rect', None)
        if rect is not None:
            x1, y1, x2, y2 = (int(v) for v in rect)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), RECT, 1)

        parts = []
        fs = getattr(region, '_drawn_font_size', None)
        if fs is not None:
            parts.append(f"{fs:.0f}px")
        fit = getattr(region, '_bubble_fit', None)
        if fit is not None:
            parts.append(f"{fit * 100:.0f}%")
        if getattr(region, '_bubble_floored', False):
            parts.append("FLOOR")
        if not parts:
            continue
        label = ' '.join(parts)

        if poly is not None and len(poly) >= 3:
            ax, ay = int(poly[:, 0].min()), int(poly[:, 1].min())
        elif rect is not None:
            ax, ay = int(rect[0]), int(rect[1])
        else:
            continue
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(ay, th + 4)
        cv2.rectangle(canvas, (ax, ty - th - 4), (ax + tw + 4, ty), color, -1)
        cv2.putText(canvas, label, (ax + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas
