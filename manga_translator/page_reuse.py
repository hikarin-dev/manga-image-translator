"""One gallery page's reuse plan and capture.

A client may send a page's earlier data (page_data) with `from` — the first stage that must run —
and `keep` — later stages whose earlier output may be kept when their inputs come out identical.
Stages before `from` are restored from the data; the rest run and are captured. The page's
complete data goes back to the client with its result.
"""
import numpy as np

from . import page_data
from .page_data import ORDER, STAGES
from .stages import IMAGE_TRANSLATORS, translator_keys
from .utils.textblock import TextBlock


def _joined(record, item):
    if 'text' in item:
        return item['text']
    texts = [record['lines'][i]['text'] for i in item['lines']]
    return TextBlock(np.zeros((len(texts), 4, 2), np.int32), texts).text if texts else ''


class PageRun:
    def __init__(self, data, config, capture=True):
        self.start = 'prepare'
        self.capture = capture                 # False: nothing is returned, so nothing is encoded
        self.keep = frozenset()
        self.prior, self.prior_blobs = None, {}
        self.rejected = None                   # why the sent data was ignored, if it was
        self.decisions = {}                    # stage -> 'ran' | 'reused'
        self.image_translator = any(key in IMAGE_TRANSLATORS for key in translator_keys(config))
        # capture
        self.detection = None
        self.recognized = []
        self.merged_regions = None
        self.items = None
        self.lines, self.read = [], []
        self.bubble_items = None
        self.end = None
        self.raw_bytes = None
        self.text, self.text_bytes = None, None
        if data:
            try:
                record, blobs = page_data.unpack(data)
                start = record.pop('from', None)
                keep = record.pop('keep', None) or []
                if start not in STAGES or ORDER[start] < ORDER['detect']:
                    raise ValueError('invalid from')
                if not isinstance(keep, list) or any(stage not in page_data.KEEPABLE for stage in keep):
                    raise ValueError('invalid keep')
                self._check(record, start)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                self.rejected = f'invalid page data: {exc}'
            else:
                self.prior, self.prior_blobs, self.start, self.keep = record, blobs, start, frozenset(keep)

    @staticmethod
    def _check(record, start):
        """The data must supply every stage it asks to skip (cheap structural checks)."""
        lines = record.get('lines')
        if not isinstance(lines, list):
            raise ValueError('lines missing')
        n = len(lines)
        if ORDER[start] > ORDER['ocr']:
            if any(not 0 <= i < n or 'text' not in lines[i] for i in record.get('read') or []):
                raise ValueError('read refers to unknown lines')
        if ORDER[start] > ORDER['merge']:
            for item in record.get('regions') or []:
                if not item['lines'] or any(not 0 <= i < n for i in item['lines']):
                    raise ValueError('region refers to unknown lines')

    def restores(self, stage):
        return self.prior is not None and ORDER[stage] < ORDER[self.start]

    def _keeps(self, stage):
        return self.prior is not None and stage in self.keep and ORDER[stage] >= ORDER[self.start]

    def reused(self, stage):
        self.decisions[stage] = 'reused'

    def ran(self, stage):
        self.decisions[stage] = 'ran'

    # ── restore ──────────────────────────────────────────────────────────────────────────────
    def restore_detect(self, decode_raw=True):
        """(lines, raw mask) as detection produced them; the mask only when something reads it."""
        self.raw_bytes = self.prior_blobs.get('raw')
        raw = page_data.decode_mask(self.raw_bytes) if self.raw_bytes and decode_raw else None
        self.reused('detect')
        return page_data.detect_lines(self.prior), raw

    def restore_ocr(self):
        self.reused('ocr')
        return page_data.ocr_lines(self.prior)

    def restore_merge(self):
        self.reused('merge')
        return page_data.regions(self.prior)

    def restore_translation(self, regions, config):
        """Survivors, when translation is restored or its earlier output still applies."""
        if self.restores('translate') and page_data.translated(self.prior):
            self.reused('translate')
            return page_data.apply_translations(self.prior, regions, config)
        if (self._keeps('translate') and not self.image_translator and page_data.translated(self.prior)
                and len(regions) == len(self.prior['regions'])
                and [r.text for r in regions] == [_joined(self.prior, item) for item in self.prior['regions']]):
            self.reused('translate')
            return page_data.apply_translations(self.prior, regions, config)
        return None

    def restore_mask(self, kept):
        """The refined mask, when it is restored or its inputs are unchanged."""
        data = self.prior_blobs.get('text')
        if not data:
            return None
        if self.restores('mask') or (self._keeps('mask') and self.restores('detect')
                                     and page_data.survivors(self.prior) == page_data.region_points(kept)):
            self.text_bytes = data
            self.reused('mask')
            return page_data.decode_mask(data)
        return None

    def restore_bubbles(self, shape):
        items = self.prior.get('bubbles') if self.prior is not None else None
        if items is None or not (self.restores('bubbles') or self._keeps('bubbles')):
            return None
        self.bubble_items = items
        self.reused('bubbles')
        return page_data.decode_bubbles(items, shape)

    # ── capture ──────────────────────────────────────────────────────────────────────────────
    def detected(self, textlines, raw):
        """(Run on the CPU pool: a new raw mask is encoded here, while other pages detect, rather
        than at the end of the page.)"""
        self.detection = page_data.detected(textlines)
        if 'detect' not in self.decisions:
            self.ran('detect')
            if self.capture and raw is not None and self.detection:
                self.raw_bytes = page_data.encode_mask(raw)

    def ocr_done(self, textlines):
        if 'ocr' not in self.decisions:
            self.ran('ocr')
        self.recognized = list(textlines)
        self.lines, self.read = page_data.encode_lines(self.detection or [], self.recognized)

    def merged(self, regions):
        if 'merge' not in self.decisions:
            self.ran('merge')
        self.merged_regions = list(regions)
        self.items = page_data.encode_regions(self.merged_regions, self.recognized)
        for index, region in enumerate(self.merged_regions):
            region._region_id = index

    def translated(self, kept):
        """Record translations. A translator that replaced the regions leaves nothing reusable."""
        if 'translate' not in self.decisions:
            self.ran('translate')
        known = {id(r) for r in self.merged_regions or []}
        if all(id(r) in known for r in kept):
            page_data.encode_translations(self.items, self.merged_regions, kept)

    def masked(self, mask):
        if self.text_bytes is None:
            self.ran('mask')
            self.text = mask

    def bubbled(self, masks):
        if 'bubbles' not in self.decisions:
            self.ran('bubbles')
            self.bubble_items = page_data.encode_bubbles(masks)

    def ended(self, stage):
        self.end = stage

    def result(self):
        """The page's complete data (run on the CPU pool: it may encode masks)."""
        record = {}
        if self.end:
            record['end'] = self.end
        record['lines'] = self.lines if self.recognized or self.lines else [
            {'pts': page_data._points(pts), 'score': page_data._score(score)} for pts, score, _ in self.detection or []]
        record['read'] = self.read
        if self.items is not None:
            record['regions'] = self.items
        if self.bubble_items is not None:
            record['bubbles'] = self.bubble_items
        blobs = {}
        if self.detection and self.raw_bytes:
            blobs['raw'] = self.raw_bytes
        text = self.text_bytes or (page_data.encode_mask(self.text) if self.text is not None else None)
        if text:
            blobs['text'] = text
        return page_data.pack(record, blobs)
