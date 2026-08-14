# Deploy FontSense on Vercel

This deployment reuses the existing Gradio interface and frozen CNN. It does
not train a model or require the generated dataset.

Use a GitHub import or the Vercel CLI. A static-site or drag-and-drop upload
does not build the Python FastAPI function and will return Vercel's `404:
NOT_FOUND` page.

## Deploy from GitHub

1. In Vercel, choose **Add New > Project**.
2. Import `rustamovalixan04-cyber/fontsense`.
3. Keep the repository root as the project root.
4. Use the **FastAPI** framework preset. The checked-in `vercel.json` also
   selects it explicitly.
5. Do not add a build command or output directory.
6. Deploy. `vercel.json` installs the CPU-only inference dependencies, and
   Vercel serves the mounted Gradio app through the FastAPI entry point.
7. Open `/healthz` on the deployment URL. A successful response has
   `"status": "ok"` and the frozen checkpoint SHA-256.

No secret or token is required by FontSense itself.

## Deploy the local dist folder

After installing and signing in to the Vercel CLI, open a terminal in the
repository and run:

```text
vercel --cwd dist
```

Use `vercel --prod` from that folder only after the preview URL works. Do not
upload the ZIP through a static-site uploader.

## Important Vercel setting

PyTorch makes this a large Python function. New Vercel projects created after
30 June 2026 are enrolled in Large Functions automatically. If an older Vercel
project reports a function-size error, enable Fluid Compute and add this
environment variable in the Vercel project settings before redeploying:

```text
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
```

## What is deployed

- `api/index.py` mounts the existing Gradio app as an ASGI application.
- `requirements-vercel.txt` contains only CPU inference and web dependencies.
- `.vercelignore` leaves training data, notebooks, tests, reports, caches, and
  the local virtual environment out of the deployment upload.
- The frozen checkpoint and its freeze record remain included and are verified
  when the process starts.

## Limits

- The first request after an idle period may be slow because PyTorch and the
  checkpoint must load in a new serverless process.
- Uploaded requests must fit Vercel's request limit. The mounted app caps files
  at 4 MB.
- Vercel's Python runtime and Large Functions support are currently beta
  features, so Colab or local Gradio remains the simpler backup demo route.
