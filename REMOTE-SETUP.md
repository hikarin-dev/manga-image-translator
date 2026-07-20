# Remote hosting — Cloudflare Tunnel setup (Phase 0/1)

Turns the local translation server into an HTTPS endpoint your friends can use from
the Shiori app, without opening any router port. The server code already enforces the
security model (see "What the server enforces" below); this file is the one-time
infrastructure setup you do by hand.

## What the server enforces (already in code, nothing to configure)

- **Worker isolation**: the GPU worker on port 5004 binds `127.0.0.1` only and is never
  reachable through the tunnel.
- **External-request gate** (`server/edge.py`): any request that arrives through the
  tunnel (or from a non-loopback address) can only reach
  `/translate/gallery/start|poll|cancel`, `/stats`, and a plain-text `/` ping.
  Everything else — `/register`, `/simple_execute/*`, `/results*`, `/queue-size`,
  `/reset-context`, `/docs`, the legacy translate endpoints, the result file mount —
  answers 404 externally and keeps working locally.
- **Access token**: when `MT_ACCESS_TOKEN` is set, external requests must send it in an
  `X-Access-Token` header (the app's Settings → Translation → "Server access token"
  field). Local requests never need it.
- **Input limits (external only)**: 150 pages/job, 8MB/page, 120MB/request, uploads must
  actually be images. The app splits >80MB galleries into multiple requests automatically
  (Cloudflare caps request bodies at 100MB).
- **Rate limits (external only)**: 2 concurrent jobs and 12 job starts/hour per IP;
  new jobs are refused with a friendly message when 8 jobs are already queued.
- **CORS**: restricted to `https://hikarin-dev.github.io` + localhost dev origins.

All limits are env-tunable — see the header of `server/edge.py`.

## One-time setup

### 1. Generate the access token

Add to `.env` in this folder (same file as the API keys):

```
MT_ACCESS_TOKEN=<paste a long random string>
```

Generate one: `powershell -c "[guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')"`

### 2. Install cloudflared and create the tunnel

Your domain must already be added to your (free) Cloudflare account.

```
winget install --id Cloudflare.cloudflared
cloudflared tunnel login                       # opens browser, pick your domain
cloudflared tunnel create shiori-translate     # prints the tunnel UUID
cloudflared tunnel route dns shiori-translate translate.YOURDOMAIN.com
```

Create `%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: shiori-translate
credentials-file: C:\Users\Admin\.cloudflared\<TUNNEL-UUID>.json
ingress:
  - hostname: translate.YOURDOMAIN.com
    service: http://127.0.0.1:5003
  - service: http_status:404
```

### 3. Add the Cloudflare rate-limiting rule (the proxy-level limiter)

Dashboard → your domain → **Security → WAF → Rate limiting rules** (1 rule free):

- If incoming requests match: Hostname equals `translate.YOURDOMAIN.com`
- Rate: **30 requests / 10 seconds** per IP  (normal use is ~1 poll per 3s per reader)
- Action: Block, for the default duration

This is the burst backstop; the in-app limits above handle the sustained/GPU side.

### 4. Run it

- `run-remote-server.bat` — the server with auto-restart on crash
  (start-translator.bat still works for local-only sessions).
- `run-tunnel.bat` — the tunnel with auto-restart.
- `install-autostart.bat` — optional: starts both, minimized, at every logon
  (Startup-folder launcher, no admin; remove with `uninstall-autostart.bat`).

## What you and your friends put in the app

Settings → Translation (app at https://hikarin-dev.github.io/shiori/app/):

- **Translation server**: `https://translate.YOURDOMAIN.com`
- **Server access token**: the `MT_ACCESS_TOKEN` value

Your own machine can keep `http://127.0.0.1:5003` with no token — local use is
unrestricted and skips the tunnel round-trip.

## Verifying after setup

```
curl https://translate.YOURDOMAIN.com/                      → "Shiori translation server is running."
curl https://translate.YOURDOMAIN.com/docs                  → 404 (gate works)
curl -X POST https://translate.YOURDOMAIN.com/translate/gallery/poll -F job_token=x -F since=0
                                                            → 401 without X-Access-Token
curl https://translate.YOURDOMAIN.com/stats -H "X-Access-Token: <token>"
                                                            → queue/today/GPU JSON
```

Then translate a small gallery from the app with the tunnel URL + token set.

## Monitoring

- `GET /stats` (token-gated externally, open locally): queue depth, workers, today's
  jobs/pages/cost, live GPU utilization, last 20 job summaries.
- `logs\jobs.jsonl` — one JSON line per finished job (pages, wall time, per-stage
  timings, GPU/VRAM peaks, LLM cost, client IP). Survives restarts.
- `logs\server-restarts.log` — every (re)start from run-remote-server.bat.

## Operator dashboard

Pool state, queue depth, GPU meters, today's totals and recent jobs, at `/dashboard`.

- **Locally: always on**, no configuration — `http://127.0.0.1:5003/dashboard`. (5003 is the
  server; 5004 is its internal worker and has no dashboard.)
- **Through the tunnel: address-gated**, at `https://translate.YOURDOMAIN.com/dashboard`.
  A browser navigation cannot send `X-Access-Token`, so the caller's address is the
  credential instead. Allowed callers are:
  - anything in `MT_DASHBOARD_IPS` — comma-separated, exact addresses or CIDR, e.g.
    `MT_DASHBOARD_IPS=203.0.113.7,198.51.100.0/24`
  - **automatically, every aux node currently connected**, so whoever is lending a GPU can
    watch the pool without being added to a list. Access lapses when their node disconnects.

  Anyone else gets a 404 — not a 403 — so the page's existence isn't advertised.

This allowlist is only trustworthy because the tunnel is the sole way in: with no forwarded
port, every external request arrives via cloudflared and `cf-connecting-ip` is stamped by
Cloudflare rather than by the caller. If you ever expose the origin directly, that stops being
true and the dashboard needs real authentication.

Two things to keep in mind. The address is the *only* check, so everyone behind the same public
address — the rest of the household, others in the same CGNAT pool — can also read it, and it
shows the aux roster plus a job history containing client IPs. And home addresses are dynamic:
if yours changes you are locked out until you update `MT_DASHBOARD_IPS`, which is why browsing
from the server machine itself should just use `127.0.0.1`.

## Adding GPU capacity: auxiliary nodes

Any spare machine can lend its GPU to this server. The node **dials out** to the server and
waits for work, so it needs no tunnel, no port forward, and no inbound rule of its own —
only outbound HTTPS. Nothing on the node listens publicly.

### On this (main) server

Add a join secret to `.env` — a *different* secret from `MT_ACCESS_TOKEN`, since this one
grants "receive other people's pages", not "submit a translation":

```
MT_AUX_TOKEN=<paste a long random string>
```

Aux joining is off entirely while this is unset. Restart the server, then confirm with
`curl http://127.0.0.1:5003/aux/nodes` (local-only) once a node connects.

### On each aux machine

Same repo, same models, same `venv`. Edit `run-aux-node.bat` to set `MT_AUX_JOIN` (your
`https://translate.YOURDOMAIN.com`) and `MT_AUX_TOKEN`, then run it. That's the whole setup —
repeat verbatim for every extra machine. Equivalent by hand:

```
python server/main.py --aux --join https://translate.YOURDOMAIN.com --aux-token <secret> --use-gpu
```

The node prints one line per chunk and nothing else: no OCR/translation chatter, and never
the `--verbose` intermediate images. Full worker output lands in `logs\aux-worker.log`.

### How work is shared

- Aux nodes are **preferred over the main server's own GPU** (`MT_AUX_PRIORITY`, default 10
  vs. local 100), so remote capacity is spent first and this machine stays responsive.
- `--lazy` (or `MT_LAZY=1`) goes further: while *any* node is connected, galleries go to the
  nodes even when they are all busy, and the local GPU is used only when no node is connected
  at all. The local worker still starts — it is that fallback — but with `--models-ttl` it
  unloads from VRAM while idle, so the card is genuinely free. Single-image requests still run
  locally either way, since nodes are gallery-only.
- A gallery is split across every free node rather than pinned to one, so two nodes really do
  halve a single gallery instead of only helping when two galleries are queued.
- If a node drops mid-chunk, only the pages it never delivered are re-queued elsewhere; the
  job continues. A node that repeatedly delivers nothing is given up on after
  `MAX_CHUNK_STALLS` attempts and the job reports a normal failure.
- Nodes are **gallery-only**. The single-image endpoints hand back a pickled object, and the
  main server won't deserialize that from a machine it doesn't own.
- A node is refused if its join token, protocol version, or code version doesn't match
  (`git rev-parse --short HEAD`) — mixed versions would render pages inconsistently *within
  one gallery*. Set `MT_AUX_ALLOW_VERSION_SKEW=1` to override during a rolling upgrade.

### Trust

An aux node sees every page it translates. Only give the join token to machines you'd hand
the images to anyway. Frames coming back are deserialized through a restricted unpickler
(`server/safe_pickle.py`), so a compromised node can't execute code on the main server, but
it can still read what it is sent.

## Known limits / later phases

- No per-user identity yet — the token is shared. Google sign-in + per-user quotas are
  Phase 2 in `plans/SAAS-PLAN.md`.
- A friend's translator config is trusted as-is (big detection/inpainting sizes make
  slow jobs, bounded by the rate limits). Tighten in Phase 2 if it becomes a problem.
- If the tunnel hostname changes, update both the app setting and the WAF rule.
