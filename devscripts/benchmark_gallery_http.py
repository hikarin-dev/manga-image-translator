"""Run isolated gallery jobs using cached originals; never writes browser libraries."""
import argparse
import asyncio
import json
import hashlib
import time
import uuid
from pathlib import Path

import aiohttp
import psutil


async def main(args):
    rows = json.loads(Path(args.index).read_text())
    groups = {}
    for row in rows:
        groups.setdefault(row['gallery'], []).append(row)
    output = Path(args.output)
    done = set()
    if output.exists():
        done = {(r['gallery'], r['renderer']) for r in map(json.loads, output.read_text().splitlines())
                if not r['terminal'].get('failed') and r['job'].get('emitted') == r['job'].get('pages')
                and not r.get('validation', {}).get('problems')}
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        async with client.get(args.server + '/dashboard/data') as response:
            initial = await response.json()
        if initial['queue']['live_jobs']:
            raise RuntimeError('Another gallery is active; benchmark requires an idle server')
        for gallery, pages in groups.items():
            if args.gallery and gallery not in args.gallery:
                continue
            m = json.loads(Path(pages[0]['manifest']).read_text('utf-8'))
            config = m['config']
            config['study_mode_generation'] = 'text_and_image'
            for renderer in args.renderers:
                if (gallery, renderer) in done:
                    continue
                if args.guard_server:
                    async with client.get(args.guard_server + '/dashboard/data') as response:
                        guard = await response.json()
                    if guard['queue']['live_jobs']:
                        raise RuntimeError('Production translation activity would contaminate the benchmark')
                config['render']['renderer'] = renderer
                token = uuid.uuid4().hex
                form = aiohttp.FormData()
                form.add_field('config', json.dumps(config))
                form.add_field('job_token', token)
                form.add_field('batch_size', str(args.batch_size))
                for row in pages:
                    raw = (Path('snapshots/objects') / row['sha256']).read_bytes()
                    form.add_field('image', raw, filename=str(row['page']) + '.image', content_type='application/octet-stream')
                start = time.perf_counter()
                available_ram_min = psutil.virtual_memory().available
                output.with_suffix('.active.json').write_text(json.dumps({'token': token, 'gallery': gallery, 'renderer': renderer}))
                async with client.post(args.server + '/translate/gallery/start', data=form) as response:
                    response.raise_for_status()
                    started = await response.json()
                print(json.dumps({'started': gallery, 'renderer': renderer, 'token': token[:8], 'pages': len(pages)}), flush=True)
                cursor, previous_done, refs, terminal = 0, -1, {}, None
                guard_at = time.perf_counter()
                try:
                    while terminal is None:
                        await asyncio.sleep(1)
                        available_ram_min = min(available_ram_min, psutil.virtual_memory().available)
                        async with client.post(args.server + '/translate/gallery/poll', data={'job_token': token, 'since': str(cursor)}) as response:
                            response.raise_for_status()
                            raw = await response.read()
                        offset = 0
                        while offset < len(raw):
                            status = raw[offset]
                            size = int.from_bytes(raw[offset+1:offset+5], 'big')
                            data = raw[offset+5:offset+5+size]
                            offset += 5 + size
                            if status == 7:
                                meta = json.loads(data)
                                cursor = meta['cursor']
                                if meta['done'] != previous_done:
                                    print(json.dumps({'gallery': gallery, 'renderer': renderer, 'done': meta['done'], 'state': meta['state']}), flush=True)
                                    previous_done = meta['done']
                                if meta['status'] in ('notfound', 'cancelled'):
                                    raise RuntimeError('Job ' + meta['status'])
                            elif status == 6:
                                b = 1 + data[0]
                                idx = int.from_bytes(data[b:b+4], 'big')
                                # Image-bearing study payloads are not plain JSON; the
                                # preceding snapshot-only metadata frame always is.
                                if data[b+4:b+5] == b'{':
                                    try:
                                        ref = json.loads(data[b+4:]).get('snapshot')
                                        if ref:
                                            refs[idx] = ref
                                    except (ValueError, UnicodeError):
                                        pass
                            elif status == 0:
                                terminal = json.loads(data)
                            elif status == 2:
                                raise RuntimeError(data.decode('utf-8', 'replace'))
                        if time.perf_counter() - start > 1800:
                            raise TimeoutError('Gallery exceeded 30 minutes')
                        if args.guard_server and time.perf_counter() - guard_at > 10:
                            async with client.get(args.guard_server + '/dashboard/data') as response:
                                guard = await response.json()
                            if guard['queue']['live_jobs']:
                                raise RuntimeError('Production activity started; cancelling only the benchmark job')
                            guard_at = time.perf_counter()
                except BaseException:
                    await client.post(args.server + '/translate/gallery/cancel', data={'job_token': token})
                    raise
                async with client.get(args.server + '/dashboard/data') as response:
                    dashboard = await response.json()
                record = next((r for r in dashboard['recent_jobs'] if r['token'] == token[:8]), {})
                record = {k:v for k,v in record.items() if k not in ('ip', 'key', 'source_url')}
                validation = {'snapshots': len(refs), 'rendered_pages': 0, 'pass_through_pages': 0, 'problems': []}
                for idx, ref in refs.items():
                    evidence = (Path('snapshots/runs') / (ref['run_id'] + '.json')).read_bytes()
                    snapshot = json.loads(evidence)
                    if hashlib.sha256(evidence).hexdigest() != ref['manifest_sha256']:
                        validation['problems'].append({'page_index': idx, 'reason': 'manifest hash mismatch'})
                    complete = [c['renderer'] for c in snapshot['render_calls'] if c['status']=='complete']
                    if renderer in complete:
                        validation['rendered_pages'] += 1
                    elif complete == ['none']:
                        validation['pass_through_pages'] += 1
                    else:
                        validation['problems'].append({'page_index': idx, 'actual_renderers': complete})
                result = dict(gallery=gallery, renderer=renderer, config=config, batch_size=args.batch_size,
                              available_ram_min_gib=available_ram_min / 1024**3,
                              client_wall_s=time.perf_counter()-start, terminal=terminal,
                              snapshot_refs=refs, job=record, validation=validation)
                with output.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result) + '\n')
                print(json.dumps({'finished': gallery, 'renderer': renderer, 'stats': record}), flush=True)
                output.with_suffix('.active.json').unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', default='dev/performance-20260922/index.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--server', default='http://127.0.0.1:5003')
    parser.add_argument('--guard-server')
    parser.add_argument('--batch-size', type=int, default=10, help='Match the app scheduling cap (DeepSeek default: 10)')
    parser.add_argument('--renderers', nargs='+', default=['manga2eng', 'shiori', 'shiori_v2'])
    parser.add_argument('--gallery', nargs='+')
    asyncio.run(main(parser.parse_args()))
