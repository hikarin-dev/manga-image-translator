'use strict';
const base = '/dashboard/feedback';
const $ = id => document.getElementById(id);
const surfaces = { original: 'Original image', ocr: 'OCR text', translation: 'Translation' };
const categories = { placement: 'Placement / alignment', readability: 'Font size / readability', line_breaks: 'Line breaks / hyphenation',
  overflow: 'Overflow / overlap', grouping: 'Grouping', ocr: 'OCR / text recognition', translation: 'Translation', other: 'Other' };
let offset = 0, current = null, generation = 0, dirty = false, saving = false;
const elem = (tag, text) => { const el = document.createElement(tag); el.textContent = text; return el; };

async function request(path, options = {}) {
  const response = await fetch(base + path, { cache: 'no-store', ...options });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function loadList() {
  $('listStatus').textContent = 'Loading…';
  try {
    const query = new URLSearchParams({ status: $('filterStatus').value, q: $('search').value, offset, limit: 40 });
    const data = await request('/list?' + query);
    $('counts').textContent = Object.entries(data.counts).map(([state, count]) => `${count} ${state}`).join(' · ');
    $('reports').replaceChildren();
    for (const report of data.reports) {
      const button = elem('button', ''); button.type = 'button'; button.dataset.id = report.report_id;
      button.classList.toggle('active', current?.report_id === report.report_id);
      button.append(elem('strong', `${surfaces[report.surface] || report.surface} · ${report.status}`),
        elem('small', new Date(report.received_at).toLocaleString()),
        elem('p', report.issues.map(i => categories[i] || i).join(', ')), elem('p', report.note || 'No user note'));
      button.onclick = () => selectReport(report.report_id);
      $('reports').append(button);
    }
    $('listStatus').textContent = `${data.total} reports${data.unreadable ? ` · ${data.unreadable} unreadable archives` : ''}`;
    $('pageCount').textContent = data.total ? `${offset + 1}–${Math.min(offset + 40, data.total)}` : '0';
    $('previous').disabled = offset === 0; $('next').disabled = offset + 40 >= data.total;
  } catch (err) { $('listStatus').textContent = 'Could not load feedback: ' + err.message; }
}

function updateEvidence() {
  if (!current) return;
  const page = current.page, variant = $('variant').value, asset = page[variant];
  const focused = page.bubbles.find(b => b.id === $('region').value);
  $('sourceText').textContent = focused?.src || 'Not captured';
  $('rawText').textContent = focused?.rawTr || 'Not captured';
  $('wrappedText').textContent = focused?.tr || 'Not captured';
  $('regionInfo').textContent = `${focused?.id || ''} · OCR lines: ${(focused?.lineIds || []).join(', ') || 'not captured'}`;
  $('highlights').replaceChildren(); $('pagePreview').hidden = !asset;
  $('previewStatus').textContent = asset ? '' : 'This image was not captured.';
  if (!asset) { $('pageImage').removeAttribute('src'); return; }
  $('pageImage').src = `${base}/${current.report_id}/asset/${asset.sha256}`;
  const selected = new Set([current.selection.primary, ...(current.selection.related || []), $('region').value]);
  for (const bubble of page.bubbles) {
    if (!selected.has(bubble.id)) continue;
    const box = variant === 'translated' ? (bubble.tbox || bubble.rbox || bubble.region || bubble.box) : bubble.box;
    if (!box || !['x', 'y', 'w', 'h'].every(k => Number.isFinite(box[k]))) continue;
    const mark = elem('div', ''); mark.className = 'region-mark';
    mark.classList.toggle('primary', bubble.id === current.selection.primary);
    mark.classList.toggle('focused', bubble.id === $('region').value);
    for (const [css, key] of [['left','x'], ['top','y'], ['width','w'], ['height','h']]) mark.style[css] = `${box[key] * 100}%`;
    $('highlights').append(mark);
  }
}

async function selectReport(id) {
  if (saving || (dirty && !confirm('Discard unsaved review changes?'))) return;
  const ticket = ++generation;
  $('reviewMessage').textContent = '';
  try {
    const data = await request('/' + encodeURIComponent(id));
    if (ticket !== generation) return;
    current = data.manifest; dirty = false;
    $('empty').hidden = true; $('detail').hidden = false;
    $('reportTitle').textContent = `Report ${current.report_id.slice(0, 12)}`;
    $('reportMeta').textContent = `${surfaces[current.selection.surface || 'translation']} · ${current.issues.map(i => categories[i] || i).join(', ')} · ${current.fidelity}`;
    $('userNote').textContent = current.note || 'No user note';
    $('gaps').textContent = current.missing.length ? 'Missing evidence: ' + current.missing.join(', ') : '';
    $('download').href = `${base}/${current.report_id}/export`;
    $('reviewStatus').value = data.review.status; $('reviewNote').value = data.review.note;
    $('displayData').textContent = JSON.stringify(current.display, null, 2);
    $('region').replaceChildren(...current.page.bubbles.map((b, i) => { const option = elem('option', `${i + 1}. ${b.id}${b.id === current.selection.primary ? ' (reported)' : ''}`); option.value = b.id; return option; }));
    $('region').value = current.selection.primary;
    $('variant').value = (current.selection.surface || 'translation') === 'translation' && current.page.translated ? 'translated' : 'original';
    updateEvidence();
    history.replaceState(null, '', `${base}?report=${encodeURIComponent(current.report_id)}`);
    document.querySelectorAll('#reports button').forEach(b => b.classList.toggle('active', b.dataset.id === current.report_id));
  } catch (err) { $('listStatus').textContent = 'Could not open report: ' + err.message; }
}

$('pageImage').onerror = () => { $('previewStatus').textContent = 'Preview unavailable. The original evidence can still be downloaded.'; };
$('variant').onchange = $('region').onchange = updateEvidence;
$('filters').onsubmit = e => { e.preventDefault(); offset = 0; loadList(); };
$('previous').onclick = () => { offset = Math.max(0, offset - 40); loadList(); };
$('next').onclick = () => { offset += 40; loadList(); };
$('reviewForm').oninput = () => { dirty = true; $('reviewMessage').textContent = ''; };
$('reviewForm').onsubmit = async e => {
  e.preventDefault(); if (!current || saving) return;
  saving = true; $('saveReview').disabled = $('reviewStatus').disabled = $('reviewNote').disabled = true;
  $('reviewMessage').textContent = 'Saving…';
  try {
    await request(`/${current.report_id}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Feedback-Review': '1' },
      body: JSON.stringify({ status: $('reviewStatus').value, note: $('reviewNote').value }) });
    dirty = false; $('reviewMessage').textContent = 'Review saved.'; await loadList();
  } catch (err) { $('reviewMessage').textContent = `Save failed (${err.message}). Your changes are still here.`; }
  finally { saving = false; $('saveReview').disabled = $('reviewStatus').disabled = $('reviewNote').disabled = false; }
};
window.addEventListener('beforeunload', e => { if (dirty || saving) { e.preventDefault(); e.returnValue = ''; } });
loadList();
const initial = new URLSearchParams(location.search).get('report');
if (initial && /^[a-f0-9]{32}$/.test(initial)) selectReport(initial);
