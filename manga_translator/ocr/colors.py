'''
Image-based text color estimation, shared by the oneocr OCR model and the
optional post-OCR color override step (config.render.estimate_font_color /
estimate_outline_color).
'''

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..utils import Quadrilateral


_KERNEL = np.ones((3, 3), np.uint8)
_MIN_SAMPLES = 8


def _median_color(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.median(crop[mask], axis=0).round().astype(int)


def _prepare_text_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    '''Returns a usable glyph mask, rejecting filled detector rectangles.'''
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = (mask[:, :, 0] if mask.shape[2] == 1
                else cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY))
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    glyphs = mask > 127
    count = int(glyphs.sum())
    coverage = count / glyphs.size
    if count < _MIN_SAMPLES or coverage >= 0.72:
        return None

    # Filled word/line boxes are not useful for separating fill, outline, and
    # background. Real glyph masks lose much more area under one erosion.
    survival = cv2.erode(glyphs.astype(np.uint8), _KERNEL).sum() / count
    if survival >= 0.84:
        return None
    return glyphs


def _otsu_text_mask(crop: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = binarized == 0
    if not dark.any() or dark.all():
        return None
    dark_u8 = dark.astype(np.uint8)
    surv_dark = cv2.erode(dark_u8, _KERNEL).sum() / int(dark.sum())
    surv_light = cv2.erode(1 - dark_u8, _KERNEL).sum() / int((~dark).sum())
    return dark if surv_dark <= surv_light else ~dark


def _dominant_distinct_color(crop: np.ndarray, zone: np.ndarray,
                             reference: np.ndarray, min_distance: int = 30
                             ) -> Tuple[Optional[np.ndarray], float]:
    '''Finds the dominant color in a zone after excluding fill-like pixels.'''
    pixels = crop[zone]
    if len(pixels) < _MIN_SAMPLES:
        return None, 0.0
    distance = np.max(np.abs(pixels.astype(np.int16) - reference.astype(np.int16)), axis=1)
    pixels = pixels[distance >= min_distance]
    if len(pixels) < _MIN_SAMPLES:
        return None, 0.0

    # Small RGB buckets resist antialiasing while preserving a real source
    # color rather than returning the center of a quantization bucket.
    buckets = pixels // 24
    _, inverse, counts = np.unique(buckets, axis=0, return_inverse=True, return_counts=True)
    winner = int(np.argmax(counts))
    members = pixels[inverse == winner]
    color = np.median(members, axis=0).round().astype(int)
    return color, len(members) / int(zone.sum())


def _is_transition_color(fg: np.ndarray, edge: np.ndarray, outer: np.ndarray) -> bool:
    '''True when edge is an antialiased blend between the fill and outer color.'''
    fg_f = fg.astype(np.float32)
    edge_f = edge.astype(np.float32)
    outer_f = outer.astype(np.float32)
    span = outer_f - fg_f
    span_sq = float(np.dot(span, span))
    if span_sq < 1.0:
        return False
    position = float(np.dot(edge_f - fg_f, span) / span_sq)
    if not 0.10 < position < 0.90:
        return False
    projected = fg_f + position * span
    residual = float(np.max(np.abs(edge_f - projected)))
    edge_distance = float(np.max(np.abs(edge_f - fg_f)))
    outer_distance = float(np.max(np.abs(outer_f - fg_f)))
    return residual <= 18.0 and outer_distance >= edge_distance + 20.0


def _ideal_outline_color(fg: np.ndarray, outline: np.ndarray) -> np.ndarray:
    '''Clamps antialiased black/white outline samples to their solid endpoint.'''
    fg = np.asarray(fg, dtype=int)
    outline = np.asarray(outline, dtype=int)
    white = np.array([255, 255, 255])
    black = np.array([0, 0, 0])

    # JPEG noise and resampling commonly turn solid white/black into a nearly
    # neutral shade. Canonicalize those before page-level color clustering.
    chroma = int(outline.max() - outline.min())
    if int(outline.min()) >= 225 and chroma <= 32:
        return white
    if int(outline.max()) <= 30 and chroma <= 20:
        return black

    # A partially covered boundary pixel is fg * alpha + outline * (1-alpha).
    # If the observed color lies on the segment from the fill to white/black,
    # recover that solid endpoint instead of rendering the blended sample.
    if _is_transition_color(fg, outline, white):
        return white
    if _is_transition_color(fg, outline, black):
        return black
    return outline


def _is_pink_fill(color: np.ndarray) -> bool:
    r, g, b = (int(channel) for channel in color)
    return r - max(g, b) >= 40 and b - g >= 15


def _normalize_outline_colors(estimates: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
    '''Recovers solid endpoints and repairs outliers using same-page style consensus.'''
    outlines = [_ideal_outline_color(fg, outline) for fg, outline in estimates]
    pink_indices = [i for i, (fg, _) in enumerate(estimates) if _is_pink_fill(fg)]
    if len(pink_indices) < 2:
        return outlines

    white = np.array([255, 255, 255])
    white_votes = sum(np.array_equal(outlines[i], white) for i in pink_indices)
    # Only override gross misses when a clear majority of the same page's pink
    # text independently resolves to white. This catches artwork/black sampled
    # by one bad mask without making a global "pink always means white" rule.
    if white_votes >= 2 and white_votes * 5 >= len(pink_indices) * 3:
        for i in pink_indices:
            outlines[i] = white.copy()
    return outlines


def _adjacent_color(crop: np.ndarray, fill_mask: np.ndarray, fg: np.ndarray) -> np.ndarray:
    '''Samples the first two pixels outside the fill, where an outline must live.'''
    fill_u8 = fill_mask.astype(np.uint8)
    dilated_once = cv2.dilate(fill_u8, _KERNEL, iterations=1).astype(bool)
    dilated_twice = cv2.dilate(fill_u8, _KERNEL, iterations=2).astype(bool)
    near_ring = dilated_once & ~fill_mask
    far_ring = dilated_twice & ~dilated_once
    near, near_support = _dominant_distinct_color(crop, near_ring, fg)
    far, far_support = _dominant_distinct_color(crop, far_ring, fg)
    if near is not None and near_support >= 0.10:
        if far is not None and far_support >= 0.10 and _is_transition_color(fg, near, far):
            return far
        return near
    if far is not None and far_support >= 0.10:
        return far
    ring = dilated_twice & ~fill_mask
    if int(ring.sum()) >= _MIN_SAMPLES:
        return _median_color(crop, ring)
    background = ~fill_mask
    if int(background.sum()) >= _MIN_SAMPLES:
        return _median_color(crop, background)
    return np.array([255, 255, 255])


def estimate_colors(crop: np.ndarray, text_mask: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Estimates text fill and outline colors from a text-line crop. When the
    detector supplies a real glyph mask, deep interior pixels estimate the fill
    and the glyph's inner boundary estimates a distinct outline. If no outline
    is present, the first pixels outside the glyph provide a background-colored
    invisible outline. Without a usable mask, Otsu segmentation supplies a
    fill mask and the same immediate-ring outline estimate is used.
    '''
    glyph_mask = _prepare_text_mask(text_mask, crop.shape[:2])
    if glyph_mask is not None:
        distance = cv2.distanceTransform(glyph_mask.astype(np.uint8), cv2.DIST_L2, 3)
        inside_distance = distance[glyph_mask]
        core_limit = max(1.0, float(np.percentile(inside_distance, 65)))
        core = glyph_mask & (distance >= core_limit)
        if int(core.sum()) < _MIN_SAMPLES:
            core = cv2.erode(glyph_mask.astype(np.uint8), _KERNEL).astype(bool)
        if int(core.sum()) < _MIN_SAMPLES:
            core = glyph_mask
        fg = _median_color(crop, core)

        # A mask may cover fill+outline or fill only. Prefer a coherent color on
        # its inner edge in the former case, then look immediately outside it.
        edge_limit = max(1.0, float(np.percentile(inside_distance, 40)))
        inner_edge = glyph_mask & (distance <= edge_limit)
        outline, support = _dominant_distinct_color(crop, inner_edge, fg)
        adjacent = _adjacent_color(crop, glyph_mask, fg)
        if outline is not None and support >= 0.18:
            if _is_transition_color(fg, outline, adjacent):
                return fg, _ideal_outline_color(fg, adjacent)
            return fg, _ideal_outline_color(fg, outline)
        return fg, _ideal_outline_color(fg, adjacent)

    fill_mask = _otsu_text_mask(crop)
    if fill_mask is None:
        return np.array([0, 0, 0]), np.array([255, 255, 255])
    core = cv2.erode(fill_mask.astype(np.uint8), _KERNEL).astype(bool)
    if int(core.sum()) < _MIN_SAMPLES:
        core = fill_mask
    fg = _median_color(crop, core)
    return fg, _ideal_outline_color(fg, _adjacent_color(crop, fill_mask, fg))


def _luminance(color) -> float:
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def snap_cluster_colors(colors: List[np.ndarray], tol: int = 40,
                        preserve_chroma: bool = True) -> List[np.ndarray]:
    '''
    Groups colors that lie within tol of each other (max channel difference,
    transitively) and snaps every member of a group to its darkest color for
    dark groups or its brightest for light ones, so pages that reuse one text
    color render it uniformly instead of with per-region jitter. Fill colors
    preserve chroma by default; outline callers disable that so pale antialias
    colors snap toward the solid light/dark endpoint instead.
    '''
    n = len(colors)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if max(abs(int(a) - int(b)) for a, b in zip(colors[i], colors[j])) <= tol:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = list(colors)
    for members in groups.values():
        chromas = [int(max(colors[i])) - int(min(colors[i])) for i in members]
        if preserve_chroma and max(chromas) >= 30:
            # Chromatic text: keep the most saturated estimate; the extremes by
            # luminance tend to be the most washed-out ones.
            rep = colors[members[int(np.argmax(chromas))]]
        else:
            lums = [_luminance(colors[i]) for i in members]
            if float(np.mean(lums)) < 128:
                rep = colors[members[int(np.argmin(lums))]]
            else:
                rep = colors[members[int(np.argmax(lums))]]
        for i in members:
            out[i] = rep
    return out


def apply_estimated_colors(image: np.ndarray, textlines, override_fg: bool, override_bg: bool,
                           text_mask: Optional[np.ndarray] = None):
    '''
    Post-OCR override: estimates fill/outline colors from the image for each
    region, harmonizes them across the page via snap_cluster_colors, and writes
    them onto the regions' fg (fill) and/or bg (outline) fields. A detector text
    mask is used only when it contains glyph shapes rather than filled boxes.
    '''
    quads = [q for q in textlines if isinstance(q, Quadrilateral)]
    if not quads or not (override_fg or override_bg):
        return

    # Some detectors return their raw mask at inference resolution rather than
    # page resolution. Align it before asking a page-space quadrilateral to crop
    # it; an empty mask is equivalent to having no mask guidance.
    mask_image = None
    if text_mask is not None and np.asarray(text_mask).size > 0:
        mask_image = np.asarray(text_mask)
        if mask_image.shape[:2] != image.shape[:2]:
            mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

    estimates = []
    estimated = []
    for q in quads:
        # OneOCR stashes its image-only estimate. Reuse it unless a detector
        # glyph mask can provide a more precise boundary estimate.
        est = getattr(q, 'color_estimate', None)
        has_estimate = est is not None
        if est is None or mask_image is not None:
            direction = getattr(q, 'assigned_direction', None) or q.direction
            try:
                crop = q.get_transformed_region(image, direction, 48)
            except Exception:
                crop = None
            if crop is not None and crop.size > 0:
                mask_crop = None
                if mask_image is not None:
                    try:
                        mask_crop = q.get_transformed_region(mask_image, direction, 48)
                    except Exception:
                        # Mask guidance is optional; image-only estimation is
                        # still better than aborting OCR for the entire page.
                        mask_crop = None
                est = estimate_colors(crop, mask_crop)
                has_estimate = True
        if est is None:
            # Degenerate/out-of-page quadrilateral: keep the OCR model's colors.
            est = (np.array([q.fg_r, q.fg_g, q.fg_b]),
                   np.array([q.bg_r, q.bg_g, q.bg_b]))
        estimates.append(est)
        estimated.append(has_estimate)

    usable = [i for i, has_estimate in enumerate(estimated) if has_estimate]
    if override_fg:
        fill_colors = snap_cluster_colors([estimates[i][0] for i in usable])
        for i, fg in zip(usable, fill_colors):
            q = quads[i]
            q.fg_r, q.fg_g, q.fg_b = int(fg[0]), int(fg[1]), int(fg[2])
    if override_bg:
        solid_outlines = _normalize_outline_colors([estimates[i] for i in usable])
        outline_colors = snap_cluster_colors(solid_outlines, preserve_chroma=False)
        for i, bg in zip(usable, outline_colors):
            q = quads[i]
            q.bg_r, q.bg_g, q.bg_b = int(bg[0]), int(bg[1]), int(bg[2])
