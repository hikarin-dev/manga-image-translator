"""Summarize measured controls and live jobs; keep the raw per-page evidence."""
import csv
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path('dev/performance-20260922')
RENDERERS = ['manga2eng', 'shiori', 'shiori_v2']


def read(name):
    p = ROOT / name
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def main():
    before, after = read('before.jsonl'), read('after.jsonl')
    key = lambda r: (r['gallery'], r['page'], r['renderer'], r['capture'])
    control = {key(r): r for r in before}
    assert len(control) == len(before) == len(after) == 271 * 3 * 2
    assert set(control) == {key(r) for r in after}
    galleries = list(dict.fromkeys(r['gallery'] for r in before))
    groups = []
    for gallery in galleries + ['all']:
        for renderer in RENDERERS:
            def select(rows, capture):
                return [r for r in rows if r['renderer']==renderer and r['capture']==capture
                        and (gallery=='all' or r['gallery']==gallery)]
            b, a = select(before, True), select(after, True)
            plain = select(before, False)
            old, new = statistics.mean(r['total_s'] for r in b), statistics.mean(r['total_s'] for r in a)
            groups.append(dict(gallery=gallery,renderer=renderer,pages=len(a),text_pages=sum(r['regions']>0 for r in a),
                layout_paint_mean_s=statistics.mean(r['total_s'] for r in plain),
                layout_paint_text_mean_s=statistics.mean(r['total_s'] for r in plain if r['regions']),
                before_mean_s=old,after_mean_s=new,reduction_percent=100*(1-new/old),
                after_p50_s=float(np.percentile([r['total_s'] for r in a],50)),
                after_p95_s=float(np.percentile([r['total_s'] for r in a],95)),
                before_snapshot_s=old-statistics.mean(r['total_s'] for r in plain),
                after_snapshot_s=new-statistics.mean(r['total_s'] for r in select(after,False)),
                after_persist_s=statistics.mean(r['persist_s'] for r in a),
                after_archive_mb=statistics.mean(r['archive_bytes']/1e6 for r in a),
                layout_mismatches=sum(r['layout']!=control[key(r)]['layout'] for r in a),
                pixel_hash_mismatches=sum(r['pixels_sha256']!=control[key(r)]['pixels_sha256'] for r in a)))
    integrity = {renderer:dict(comparisons=sum(r['renderer']==renderer for r in after),
        layout_mismatches=sum(r['renderer']==renderer and r['layout']!=control[key(r)]['layout'] for r in after),
        pixel_hash_mismatches=sum(r['renderer']==renderer and r['pixels_sha256']!=control[key(r)]['pixels_sha256'] for r in after)) for renderer in RENDERERS}
    attempts = read('live-after.jsonl')
    live = list({(r['gallery'],r['renderer']): r for r in attempts
                 if r['job'].get('emitted') == r['job'].get('pages') and not r['terminal'].get('failed')
                 and not r.get('validation',{}).get('problems')}.values())
    summary = dict(controls=groups, integrity=integrity,
                   live=[{k:r[k] for k in ('gallery','renderer','client_wall_s','job','terminal','validation','available_ram_min_gib')} for r in live],
                   live_attempts=len(attempts),
                   live_before=[{k:r[k] for k in ('gallery','renderer','client_wall_s','job','terminal','validation')} for r in read('live-before.jsonl')])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (ROOT/'controls.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(groups[0])); writer.writeheader(); writer.writerows(groups)
    print(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#3f77b6','#319782','#ba7535']
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    labels=['manga2eng','Shiori','Hybrid']
    for ax,gallery in zip(axes,galleries):
        rows=[r for r in groups if r['gallery']==gallery]
        x=np.arange(3)
        ax.bar(x-.18,[r['before_mean_s'] for r in rows],.36,color='#b9bfc7',label='Before')
        ax.bar(x+.18,[r['after_mean_s'] for r in rows],.36,color=colors,label='Optimized')
        ax.set_xticks(x,labels); ax.set_title(gallery+'\n'+str(rows[0]['pages'])+' pages')
        ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('Seconds/page\nRender + capture + durable storage')
    axes[0].legend(); fig.suptitle('Fixed inputs, normalized RGB; OCR, translation, segmentation and transport excluded')
    fig.tight_layout(); fig.savefig(ROOT/'capture-performance.png',dpi=160); plt.close(fig)
    if len(live)==9:
        fig,ax=plt.subplots(figsize=(10,4.5)); x=np.arange(3)
        for i,renderer in enumerate(RENDERERS):
            values=[next(r['job']['wall_s']/r['job']['pages'] for r in live if r['renderer']==renderer and r['gallery']==g) for g in galleries]
            bars=ax.bar(x+(i-1)*.24,values,.24,color=colors[i],label=labels[i])
            ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.axhspan(1.5,2.0,color='#aab4c0',alpha=.18,label='Requested 1.5–2 s/page range')
        ax.set_xticks(x,galleries); ax.set_ylabel('Full-pipeline seconds/page'); ax.set_title('Real cached originals through the local HTTP gallery API')
        ax.legend(ncol=1,loc='upper right',fontsize=9); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
        fig.tight_layout(); fig.savefig(ROOT/'pipeline-performance.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
