# Optional OpenAI typography analysis

The independent private `/analyze` API can send a prepared text crop to OpenAI for typography
observations, visually similar font suggestions, pairing suggestions, and
readability advice. This is a separate optional product feature. The frozen
CNN, its five probabilities, preprocessing, threshold, local Gradio demo,
Colab notebook, Windows build, and assessed evaluation remain unchanged.

## Model verification

On 1 November 2026, OpenAI's public API catalog and the dedicated model page
listed `gpt-6.1-sol`, image input, the Responses API, and Structured Outputs.
The implementation uses that exact API model ID. A model name shown in
ChatGPT alone would not establish API access. Public documentation also does
not prove access, quota, or billing availability for a particular API project.

Official sources:

- [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
- [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [List available models](https://developers.openai.com/api/reference/resources/models/methods/list)

`OPENAI_VISION_MODEL` is a server-only override for a model that your API
project can actually use. Changing it requires checking image input,
Structured Outputs and `reasoning.effort=low` compatibility first. There is
no silent model substitution or fallback to a different provider.

## Enable the private pilot

Set these variables in the Vercel project environment, then redeploy:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | A restricted key for your OpenAI API project. Store as a Secret. |
| `FONTSENSE_ANALYSIS_ACCESS_CODE` | A separate random private pilot code of at least 32 characters. Store as a Secret. |
| `UPSTASH_REDIS_REST_URL` | The HTTPS REST URL of a dedicated Upstash Redis database. |
| `UPSTASH_REDIS_REST_TOKEN` | Its server-side REST token, with EVAL permission. Store as a Secret. |
| `FONTSENSE_ANALYSIS_DAILY_LIMIT` | Optional shared daily request cap, default 25, valid range 1–500. |
| `OPENAI_VISION_MODEL` | Optional; defaults to `gpt-6.1-sol`. |

Use separate production and preview credentials and databases. Do not paste
keys into chat, HTML, browser storage, Git, or URLs. `.env.example` documents
names without values. Python does not load `.env` automatically: supply the
variables in the process environment for local runs.

The simplified public page uses local font retrieval and has no paid-analysis
control. A private API client must send its pilot code in the Authorization
header over HTTPS, never an OpenAI API key. This is a small private pilot,
not a subscription or multi-user account system. Rotate the shared code if
access needs to be revoked; build individual accounts before a public paid
launch.

Analysis stays disabled if a required setting is missing, the code is too
short, the Redis URL is invalid, or the daily limit is invalid. The public
status endpoint exposes only `enabled`, never configuration values. A
provider or Redis outage fails the analysis request without affecting
category prediction.

## Request flow and limits

1. A private client explicitly requests analysis and discloses that its crop
   will be sent to OpenAI. The public font-matching page never calls this API.
2. `/analyze` verifies the private code before reading the image or calling
   either paid service. It rejects invalid types, corrupt files, uploads over
   4 MB, and images over 20 million pixels, including streamed requests.
3. The server sends a freshly encoded JPEG no larger than 1600 by 1600 pixels,
   preserving aspect ratio and removing source metadata. It never sends the
   CNN's 112 by 48 grayscale input or a client-supplied remote image URL.
4. Redis atomically reserves usage across instances: three requests in a UTC
   calendar minute and the shared cap per UTC calendar day, separated by
   `VERCEL_ENV`. Fixed-window boundaries can permit bursts around reset time.
   Reservations are consumed even if a provider call fails. This limits
   attempts, not currency; also configure the API project's spend controls.
5. One Responses request uses low reasoning effort, at most 2400 output
   tokens including reasoning, no automatic retries, and `store=False`.
   `store=False` does not promise zero retention by the provider; consult
   OpenAI's data policy for your account.
6. The server validates structured output. Refusals, incomplete responses,
   malformed output, quota exhaustion, and timeouts produce short errors.
   Upstream error text, credentials, and uploaded images are not logged by
   this integration. The complete analysis handler has a 45-second timeout.
7. A private client must render advice as text, separately from classifier or
   retrieval results, and discard stale responses when changing images.
   Canceling in the browser does not guarantee cancellation of a request
   already received by OpenAI, so it can still consume usage.

Font names are unverified visual suggestions. The app does not claim exact
font identification, checked licenses, download availability, or calibrated
probabilities from OpenAI. Unreadable crops return no font suggestions.
The public page now uses separate catalog-backed local retrieval with pinned
font licenses; see [font matching](font_matching.md). That evidence does not
validate font names generated by the private OpenAI API.

## Verification and rollout

Install the runtime and test dependencies from `requirements-vercel.txt`,
`requirements-lite.txt` and `requirements-ci.txt`, install this project with
`pip install -e .`, then run `python -m pytest -q`. The new tests use the real
OpenAI SDK with local HTTP responses; they make no billable API requests.
They cover the API contract, image encoding, access checks, usage failures,
provider errors, refusals, incomplete responses and validation failures.

Before enabling production, verify that your API project lists the chosen
model and run one small image request with its real credentials. Test an
authorized request and a wrong-code request against the deployed endpoint,
confirm quotas, and confirm `/predict` still returns the frozen CNN result.
Evaluate useful advice, latency and actual usage costs on a separate set of
clear, ambiguous, mixed-font and unreadable example crops. Do not tune against
or rerun the held-out capstone test set for this product feature.

Local HTTP fixtures establish integration behavior, not model accuracy or
live account access. No new accuracy, latency or cost benchmark is claimed.

## Files and AI assistance

- `api/analysis.py` handles configuration, access-code comparison, Redis usage
  reservations, image encoding and the provider call. Redis is required so
  restarting or adding a function instance does not reset the shared limit.
- `api/index.py` adds the independent analysis endpoints and bounded uploads.
- `api/static/index.html` uses local matching; private analysis has no public UI.
- `requirements-analysis.txt` declares the OpenAI SDK and REST client.
- `tests/test_analysis.py` verifies behavior without paid requests.

Codex assisted with this optional integration, documentation and tests.
This work does not add new experiments or alter the capstone's evaluation.
