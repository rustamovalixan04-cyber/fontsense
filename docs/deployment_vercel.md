# Deploy FontSense on Vercel

The stateless FastAPI web product accepts one image per request and returns
visually similar fonts. The original Gradio and Colab category demos remain
available separately. There is no upload session or keepalive dependency.

## Deploy

1. Import `rustamovalixan04-cyber/fontsense` into Vercel, or deploy a local
   repository checkout using the Vercel CLI.
2. Keep the repository root and FastAPI framework preset. Do not set an output
   directory or separate build command; `vercel.json` supplies installation.
3. Installation uses `requirements-vercel.txt`, then downloads and verifies
   the pinned OCR model and 56 licensed font families in 159 real styles. Outbound build access
   to the manifest source URLs is required. Runtime requests do not download assets.
4. Open `/healthz`: expect `status: ok` and original checkpoint hash
   `c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f`.
5. Upload a clear one-line crop. Confirm three matching previews load and the
   text-correction Refine flow works before promoting a preview to production.

No key is needed for matching. The separate private OpenAI pilot requires
server credentials and shared rate limits; see [its guide](openai_analysis.md).
Do not put those secrets into the browser, source files, or URLs.

The checked-in font-name encoder and reference-statistics archives are included
with the function. No name-model training or native Windows fonts are needed
during deployment. A smoke test should upload the Impact regression PNG and
confirm `identified_font.family` is `Impact` with separate alternatives.
See [font identification](font_identification.md) for coverage and limits.

## Deployment contents

The trained retrieval release also bundles `retrieval_encoder.onnx`, its
metadata, `retrieval_index.npz/json`, `retrieval_confidence.json` and
`retrieval_policy.json`. ONNX Runtime runs the small encoder on CPU; PyTorch is
only a training dependency. The API modules matching `retrieval_*.py`,
`glyph_match.py` and `font_variations.py` must be deployed with these artifacts.
The reference index contains derived native-font statistics, never native font
programs. See [training and release evidence](retrieval_training.md).

`retrieval_policy.json` enables the selected product matcher. Setting the
server-only environment variable `FONTSENSE_RETRIEVAL_ENGINE=legacy` restores
the previous exhaustive stage for diagnosis. The trained stage also falls back
to exhaustive comparisons when its best direct-shape evidence is weak.
No key or remote model request is required for this path.

Poster handling adds `api/text_regions.py`, `api/ocr_refine.py` and
`ocr_layout_source.json`. The existing asset preparation script also downloads
`text_detection.onnx` and `text_recognition_secondary.onnx`, checks their pinned
hashes and includes them through `artifacts/web/**`. They add about 12.7 MB of
model data, use the existing ONNX Runtime dependency, and do not download at
request time. Verify a mixed poster, its text-line selector and highlighted
region as well as clean-line regressions. See [poster matching](poster_matching.md).

The extended retrieval stage adds `extended_catalog.json` and
`api/extended_match.py`. Build-time fontTools creates checksum-verified, renamed
Latin reference subsets and retains each OFL license. No new network requests
or font downloads occur during matching. See
[expanded retrieval](extended_font_retrieval.md) for the search, licensing,
cache behavior and supported scope.

`.vercelignore` uploads only the API, page, dependency/config files, small
verified CNN export, asset manifests and preparation script. `includeFiles`
bundles installed font/OCR assets and licenses. Notebooks, training datasets,
historical reports, Gradio, PyTorch and torchvision are excluded from the web
runtime. The original training checkpoint remains in the repository unchanged.

The page loads independently of matching. NumPy executes the unchanged CNN;
ONNX Runtime handles one-line OCR. A verified 1.6KB learned combiner selects
the broad category; its runtime is NumPy only. Reference rendering and vectors are cached
in each running process. The browser caches repeat requests for its selected
image and cancels stale results when the user resets or changes the image.

## Limits and verification

- Hosting may still start a new process after inactivity. A lighter runtime
  reduces startup work but does not promise an always-running server.
- The browser accepts PNG/JPEG/WebP source images up to 25MB, prepares a JPEG
  with longest edge 1800px, and limits uploads to 4MB. Direct API image requests
  accept PNG/JPEG up to 4MB and 20 million pixels.
- Matching is restricted to the reference catalog. Similar-font names are
  alternatives, not verified exact identification. See [matching limitations](font_matching.md).
- Use protected preview share URLs for testing when Vercel Authentication is
  enabled. Temporary links expire; a stable public domain is separate.
- Do not use an outdated `dist` copy or a static-site uploader; the current
  Python API and prepared assets must be built together.
