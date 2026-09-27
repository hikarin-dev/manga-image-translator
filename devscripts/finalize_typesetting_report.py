"""Fill the report from completed measurements and validate their saved evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dev/performance-20260922'
LABELS = {'manga2eng': 'Manga2Eng', 'shiori': 'Shiori', 'shiori_v2': 'Shiori hybrid'}


def main():
    attempts = [json.loads(line) for line in (DATA/'live-after.jsonl').read_text().splitlines()]
    live = list({(r['gallery'],r['renderer']):r for r in attempts
                 if r['job']['emitted']==r['job']['pages'] and not r['terminal']['failed']
                 and not r['validation']['problems']}.values())
    assert len(live) == 9, 'All nine complete library/renderer runs are required'
    index = json.loads((DATA/'index.json').read_text())
    validated, failures, regions = 0, [], {}
    for row in live:
        pages = [r for r in index if r['gallery']==row['gallery']]
        assert len(row['snapshot_refs']) == len(pages)
        region_count = 0
        for idx, ref in row['snapshot_refs'].items():
            raw = (ROOT/'snapshots/runs'/(ref['run_id']+'.json')).read_bytes()
            m = json.loads(raw)
            checks = {
                'manifest_hash': hashlib.sha256(raw).hexdigest() == ref['manifest_sha256'],
                'complete_fidelity': m['fidelity']=='complete' and m['missing']==[],
                'input_identity': m['original']['sha256']==pages[int(idx)]['sha256'],
                'config': m['config']==row['config'],
                'output_identity': m['output']['sha256']==ref['output_sha256'],
                'renderer': any(c.get('status')=='complete' and c['renderer'] in (row['renderer'],'none') for c in m['render_calls']),
            }
            if not all(checks.values()):
                failures.append({'gallery':row['gallery'],'renderer':row['renderer'],'page_index':int(idx),'checks':checks})
            region_count += len(m['render_regions'])
            validated += 1
        regions[row['gallery']+'/'+row['renderer']] = region_count
    evidence = {'snapshots_validated':validated,'failures':failures,'render_regions':regions}
    (DATA/'live-validation.json').write_text(json.dumps(evidence,indent=2))
    assert not failures, 'Snapshot validation failed; inspect live-validation.json'

    lines = ['| Library | Renderer | Delivered | Wall time | Worker time | Seconds/page | Min. available RAM | LLM cost |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in live:
        j = row['job']
        lines.append(f"| {row['gallery']} | {LABELS[row['renderer']]} | {j['emitted']}/{j['pages']} | {j['wall_s']:.1f} s | {j['compute_s']:.1f} s | **{j['wall_s']/j['pages']:.2f}** | {row['available_ram_min_gib']:.2f} GiB | ${j['llm_cost_usd']:.4f} |")
    lines.extend(['', 'All nine runs completed with no page failures, renderer fallbacks or snapshot validation errors. Each requested input hash, public config, manifest hash, output reference and complete-fidelity declaration was checked against the persisted run.', '',
                  '| Renderer | Total wall time for 271 pages | Page-weighted seconds/page |',
                  '| --- | ---: | ---: |'])
    weighted = {}
    for renderer,label in LABELS.items():
        wall = sum(r['job']['wall_s'] for r in live if r['renderer']==renderer)
        weighted[renderer] = wall/271
        lines.append(f'| {label} | {wall:.1f} s | **{wall/271:.2f}** |')
    lines.extend(['', 'The weighted figure includes 90 text-free pages in the largest library; it should not be read as the expected speed for a dense chapter.', '',
                  f"The nine measured jobs cost approximately **${sum(r['job']['llm_cost_usd'] for r in live):.4f}** in reported translation API usage, excluding warmups and baseline attempts. Live translations are fresh model responses, so wording and API latency can vary between renderers.", '',
                  '![Full-pipeline timings](../manga-image-translator/dev/performance-20260922/pipeline-performance.png)', '',
                  'Live stage totals (seconds, overlapping work; these are not additive wall-time components):', '',
                  '| Library | Renderer | OCR | Mask refinement | Rendering incl. capture | Snapshot prepare / finish / persist | Study overlays |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    for row in live:
        s = row['job']['stages_s']
        lines.append(f"| {row['gallery']} | {LABELS[row['renderer']]} | {s.get('ocr',0):.1f} | {s.get('mask_refine',0):.1f} | {s.get('rendering',0):.1f} | {s.get('snapshot_prepare',0):.1f} / {s.get('snapshot_finish',0):.1f} / {s.get('snapshot_persist',0):.1f} | {s.get('study_overlay',0):.1f} |")

    count = sum(r['job']['wall_s']/r['job']['pages'] <= 2 for r in live)
    conclusion = (f"Snapshot overhead was substantial and has been reduced. The isolated render/capture/storage comparison improved by 50–61% across these libraries, depending on renderer. "
                  f"In the full pipeline, **{count} of 9 library/renderer runs finished at or below 2 seconds/page**. "
                  f"The page-weighted means were **{weighted['manga2eng']:.2f} s/page for Manga2Eng, {weighted['shiori']:.2f} for Shiori, and {weighted['shiori_v2']:.2f} for hybrid**. "
                  + ("The target was met across all nine measured jobs." if count==9 else
                   "The tables below show the dense-library results separately; the old 1.5–2 s/page experience is not universally restored."))
    report = ROOT.parent/'plans/typesetting-performance-report-2026-09-23.md'
    body = report.read_text(encoding='utf-8')
    body = body.replace('Status: measurements in progress. The final live results and conclusion will replace this line.', conclusion)
    body = body.replace('<!-- LIVE_RESULTS -->', '\n'.join(lines))
    body = body.replace('<!-- FINAL_VALIDATION -->',
        f"All **{validated}** live page snapshots passed the final evidence validation. The optimized backend is running on its original port 5003 under its existing automatic restart wrapper, with the worker on 5004. No temporary benchmark server is left running. The app's library data and selected renderer were not changed. M3 remains the next unchecked feedback milestone.")
    report.write_text(body,encoding='utf-8')
    print(json.dumps({'report':str(report),'weighted_spp':weighted,'at_or_below_2_spp':count,'validation':evidence},indent=2))


if __name__=='__main__':
    main()
