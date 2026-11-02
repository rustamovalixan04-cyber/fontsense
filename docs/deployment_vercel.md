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

## Deployment contents

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
