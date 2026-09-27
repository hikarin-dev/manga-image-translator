# Typesetting feedback (M2)

In Study mode, hover an original image bubble, its OCR text, or a revealed
translation and press **F**. The dialog identifies which surface is being reported,
freezes that page, highlights the selected region, and
offers issue categories, an optional note, and related regions for grouping reports.
Ctrl/Cmd+F remains browser Find. Repeated F keydown and editable targets are ignored;
reader navigation, zoom, and view shortcuts pause while the dialog is open.

**Send feedback** sends the page evidence and note to the translation server for
developer review. The dialog does not reveal its address, storage path, or private
review page. Other users can submit to this installation when they use its reachable
server address and access token, if required. Users running their own translation
server send reports to that server instead; there is no separate telemetry service.
Credentials are sent only to the configured server, and are never archived.

Success is shown only after the server publishes the archive. A failed or timed-out
request leaves the note and evidence available for retry. Closing and reopening the
same region retains its draft during this reader session; drafts are held in memory,
so reload or closing the tab discards unsaved drafts. Editing a submitted report
creates a new report ID. Retrying the same request is idempotent, including a retry
after a lost response. Concurrent retries cannot overwrite an acknowledged report.

**Export ZIP** downloads the acknowledged archive when available. Before sending, or
when that server is unavailable, it exports the same cached evidence and the note
instead. No report operation fetches source pages, invokes translation, or starts
acquisition work.

## Archive contract

Each archive is `feedback/<32-hex-report-id>.zip`, containing `manifest.json` and
`assets/<sha256>` entries. The versioned manifest has:

| Field | Evidence |
| --- | --- |
| `schema`, `version`, `report_id`, `created_at` | `typesetting-feedback`, version 1, generated report ID, capture time |
| `issues`, `note`, `selection` | Issue categories (including OCR), explanation, primary and related region IDs, and `surface`: `original`, `ocr`, or `translation` (older reports default to translation) |
| `page` | Page position/dimensions, cached original and translated image, clean study background, all study regions including raw/wrapped wording and available per-region glyph images |
| `display` | Reader mode, study preferences, zoom/fit/direction, viewport/DPR, page and region rectangles, reveal state, visible DOM wording and measured text styles/geometry |
| `pipeline` | The page's pipeline data as the app keeps it (lines, reading order, regions with translations, balloon masks; see `PIPELINE_DATA.md`), its masks as assets, and the translation job it belongs to |
| `translation` | That job's effective config and stage builds (no credentials or arbitrary settings) |
| `fidelity`, `missing` | Explicit evidence gaps; region IDs are indices into `pipeline.regions`, and older records without them use `legacy-<index>` |
| `request_sha256` | Durable-save request identity for safe retries |

All asset references have a fixed hash-derived path, SHA-256, byte count, and media
type. Page, study and pipeline data are captured from the same cached record. Mounted
study metadata must match that record and belong to the translation that produced the
page. The server independently checks the selected IDs against the pipeline regions and
every asset hash. A mixed translation is rejected without clearing the dialog note.
Missing original, translation, study background, and pipeline evidence are shown
explicitly. Text-only reports can be useful while remaining incomplete.

Original/OCR reports do not require a translation for the selected region. Missing
OCR wording is recorded explicitly. These reports retain the same full-page
evidence and revision checks as translation reports.

## Private review page

Open **Feedback** from the backend dashboard, or visit `/dashboard/feedback` on the
same server and port. The inbox shows saved reports, page previews with highlighted
regions, original/OCR wording, raw and displayed translations, and evidence gaps.
Filter by status, search notes/categories/report IDs, and download the evidence ZIP.
Each report can be marked **new**, **reviewed**, **resolved**, or **archived** and given
a private operator note. Archiving hides nothing from an explicit all-status search
and does not delete evidence.

Review status and notes live in `feedback/reviews/<report-id>.json`. They never modify
the submitted ZIP or appear in the user's export. Back up `feedback/` as a whole to
retain both evidence and review state.

The page, its assets, and its API are available only locally or from addresses
explicitly allowed by `MT_DASHBOARD_IPS`. Connected auxiliary workers do not inherit
feedback access. Ordinary translation tokens cannot open it, and the dashboard link
is omitted for nonoperators. The existing trusted tunnel/IP rules still apply.
Private responses are not cached, and review writes require a same-origin action
header. Submitted text is displayed as text, never executed as HTML.

## Storage and API

`POST /feedback/save` accepts one multipart file named `archive`. The server validates
entry names, sizes, hashes, and selection; the archive already carries everything, so
nothing is looked up or added. It writes and fsyncs a temporary file, then atomically
publishes it without replacing an existing report. Its acknowledgement includes `report_id`, `archive_sha256`,
`fidelity`, `missing`, and `storage: "translation-server"`.

`POST /feedback/export` accepts form fields `report_id` and `archive_sha256`, verifies
the stored archive hash, and returns the ZIP. Both routes use the existing server
access controls. Archives have a 512 MiB limit; remote requests also remain subject
to the existing configured HTTP body limit. Invalid/mismatched evidence returns
409, an unavailable report returns 404, and failed durable storage returns 507.

Reports have no automatic expiration and are outside temporary job/result cleanup.
They survive library deletion. Each report includes its own assets, so it can be
copied independently. Reports are
Git-ignored and can contain original pages and user notes; back up or delete them
explicitly as needed. The ZIP is data-only. M3 will provide the standalone replay
and comparison tool; this milestone adds no training, benchmark or layout changes.

## Validation and activation

- `npm.cmd test` includes cached-only capture, both display modes, original/OCR
  targets, deletion survival, mixed-run rejection, offline failures, identity
  handling, shortcut selection, and localized send/privacy copy.
- `venv/Scripts/python.exe -m pytest test/test_feedback.py test/test_feedback_review.py -q` checks self-contained
  storage/export, retry races, missing evidence, malformed assets, save failure, and
  remote authentication, plus private review access, previews, immutable evidence,
  source reports, and separate review state. Dashboard access integration is covered
  by `test/test_aux_pool.py`.
- Browser validation uses a disposable library and a separate test server. It covers
  image and DOM hover, related-region selection, modal shortcut isolation, a durable
  save followed by library deletion and export, offline notes/export, and stale study.
  Follow-up checks cover reporting before translation reveal, OCR text submission,
  the private inbox, highlighted previews, and persistent operator notes/status.

Restart the translation server and reload the app to activate these sources. The
isolated validation server does not activate changes in the running production
backend.
