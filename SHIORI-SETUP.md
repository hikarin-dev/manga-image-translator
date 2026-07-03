# Shiori translation server — setup notes

This local install powers the **Translate** button in the Shiori extension.

## Start the server
Double-click **`start-translator.bat`** (or run it from a terminal). Leave the window
open while translating; close it / Ctrl+C to stop. First launch after a reboot takes a
few seconds to load models onto the GPU.

**Stop the server:** close its console window (or Ctrl+C). If it was ever left running
without a window (orphaned process), run **`stop-translator.bat`** to free ports 5003/5004.

- API: `http://127.0.0.1:5003`  (set this in Shiori → Settings → Translation)
- GPU worker: auto-spawned on `127.0.0.1:5004`
- Runs on the RTX 3070 via CUDA (torch cu124).

## Shiori settings
- **Translation server:** `http://127.0.0.1:5003`
- **Translator:** `Sugoi` (offline, Japanese→English, no API key) is fast and good for
  manga. For Korean/Chinese pages choose `M2M100`/`NLLB`. For more natural,
  context-aware output choose **`Qwen2.5-7B — local LLM`** (see below).
- **Translate to:** English.

## Optional: local LLM translator (Qwen2.5-7B via Ollama)
A local LLM gives more natural translation than Sugoi. **Ollama** is installed
(auto-runs at login) with `qwen2.5:7b` pulled (~4.7 GB). Two Shiori translator options
both run on it — switch from the dropdown, no restart:

- **"Qwen2.5-7B — local LLM (fast, per-page)"** = the `custom_openai` slot. Each page is
  translated on its own (no cross-page memory). Runs 2 pages at a time.
- **"Qwen2.5-7B + cross-page context (slower, most consistent)"** = the `chatgpt` slot,
  pointed at Ollama. Feeds the previous few pages to the model for consistent names /
  pronouns / tone across a chapter. Runs pages one-at-a-time in order, and Shiori calls
  `/reset-context` at the start of each gallery so titles don't bleed into each other.

`start-translator.bat` sets all of this:
- `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` — **required**, or the LLM translators crash
  trying to log Japanese on the Windows console.
- `CUSTOM_OPENAI_MODEL=qwen2.5:7b` (custom_openai slot).
- `OPENAI_API_BASE=http://localhost:11434/v1`, `OPENAI_API_KEY=ollama`,
  `OPENAI_MODEL=qwen2.5:7b` (chatgpt slot → Ollama).
- `--context-size 4` — pages of history fed to the context mode.

Notes:
- Requires Ollama running. On 8 GB VRAM the model co-fits with the vision models (Q4);
  Ollama frees it from VRAM ~5 min after last use. Swap models with `ollama pull <name>`
  and update the env var(s).
- Local server tweaks for this: `server/args.py` + `server/main.py` forward `--context-size`
  and add a `/reset-context` endpoint; `manga_translator.py` adds `reset_page_context()`.

## Optional: Gemini (cloud) translator
Smarter than local Qwen, but cloud-based and rate-limited. Choose **"Gemini Flash"** in
Shiori's translator dropdown.
1. Get a free key at https://aistudio.google.com/apikey (no card needed).
2. Paste it into `.env` here as `GEMINI_API_KEY=...`.
3. Run `list-gemini-models.bat`, then set `GEMINI_MODEL` in `.env` to one of the listed
   ids (without the `models/` prefix), e.g. a Gemini Flash variant.
4. Restart `start-translator.bat`. Pick "Gemini Flash" in Shiori.

Caveats:
- The build's gemini translator already sets all safety filters to `BLOCK_NONE`, so adult
  content works; sexual content involving minors is blocked by Google regardless.
- **Free-tier rate limit varies a lot by model.** `gemini-3.5-flash` is only **5 req/min**
  (very slow); `gemini-2.5-flash` ≈ 15/min; a **flash-lite** is usually ≈ 30/min. For free-tier
  use, prefer a flash-lite or 2.5-flash over the newest model. Daily cap ≈ 1,500. Shiori runs
  Gemini one page at a time.
- Patched `gemini.py`: it crashed on rate-limit errors (`server_error_attempt` used before
  assignment); now it backs off 10s and retries instead of failing the page.
- **Privacy:** on the free tier Google may use your content to improve their products.
  Paid tier doesn't. Local Qwen sends nothing off-machine.
- Gemini runs per-page (no cross-page context — that's the `chatgpt`/Ollama mode only).

## Local modifications (don't `git pull` over these without re-checking)
- `requirements.txt`: `transformers` pinned to `4.46.3` (5.x needs torch ≥2.7 and breaks
  manga-ocr); `pydensecrf` commented out (needs a C++ compiler that isn't installed).
- `manga_translator/mask_refinement/text_mask_utils.py`: `refine_mask` imports pydensecrf
  lazily and falls back to the raw mask when it's absent.
- `start-translator.bat`: sets `MT_WEB_NONCE=None` — the server never forwards the
  internal nonce to its worker, so leaving it on causes a 401 "Nonce does not match".
- `fonts/ccvictoryspeech.ttf` + `manga_translator/rendering/__init__.py`: the `manga2eng`
  renderer's default font is set to CC Victory Speech (converted from the Ichigo extension's
  bundled woff2) for Ichigo-style comic lettering. To revert, point it back at
  `fonts/comic shanns 2.ttf`. After changing the renderer code, restart the server.

## Python env
Dedicated venv at `.\venv` (Python 3.11.9). To reinstall deps:
`venv\Scripts\python.exe -m pip install -r requirements.txt`
