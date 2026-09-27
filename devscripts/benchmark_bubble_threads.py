"""Measure real bubble masks with bounded ORT threading; compare exact masks."""
import argparse
import hashlib
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manga_translator.rendering.bubble_seg import _detect, MODEL_PATH


def main(args):
    rows = [r for r in json.loads((ROOT/'dev/performance-20260922/index.json').read_text()) if r['regions']]
    if args.limit:
        # Spread the quick probe across the three libraries.
        groups = [[r for r in rows if r['gallery']==g] for g in dict.fromkeys(r['gallery'] for r in rows)]
        rows = [group[i] for i in range(args.limit) for group in groups if i<len(group)][:args.limit]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = args.threads
    if args.no_spin:
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
    session = ort.InferenceSession(MODEL_PATH,options,providers=['CPUExecutionProvider'])
    def run(row):
        image = np.asarray(Image.open(io.BytesIO((ROOT/'snapshots/objects'/row['sha256']).read_bytes())).convert('RGB'))
        start = time.perf_counter()
        masks = _detect(session,image)
        elapsed = time.perf_counter()-start
        return {'gallery':row['gallery'],'page':row['page'],'seconds':elapsed,
                'mask_hashes':[hashlib.sha256(m.tobytes()).hexdigest() for m in masks]}
    run(rows[0])
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run,rows))
    output = {'threads':args.threads,'spinning':not args.no_spin,'concurrency':args.concurrency,
              'pages':len(rows),'wall_s':time.perf_counter()-start,'results':results}
    Path(args.output).write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='results'}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--concurrency',type=int,default=1)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--no-spin',action='store_true')
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
