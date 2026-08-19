# Deploy FontSense on Vercel

This deployment reuses the existing frozen CNN through a small stateless web
interface. The local and Colab versions still use Gradio. The Vercel adapter
sends each image and prediction in one request because separate serverless
requests cannot safely share Gradio's temporary upload files. It does not
train a model or require the generated dataset.

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
   Vercel serves the stateless FastAPI interface.
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

- `api/index.py` receives the uploaded image and predicts in one request.
- `api/static/index.html` provides the lightweight browser interface and
  previews an image locally before it is sent. Large PNG, JPEG, and WebP
  photos are resized and compressed in the browser to make uploads faster.
- `requirements-vercel.txt` contains only CPU inference and web dependencies.
- `.vercelignore` leaves training data, notebooks, tests, reports, caches, and
  the local virtual environment out of the deployment upload.
- The frozen checkpoint and its freeze record remain included and are verified
  when the process starts.

## Limits

- The first prediction after an idle period may be slow because PyTorch and
  the checkpoint must load in a new serverless process. The page and image
  preview do not need to wait for this model startup.
- Uploaded requests must fit Vercel's request limit. The browser accepts source
  photos up to 25 MB, then prepares an upload no larger than 4 MB.
- Vercel's Python runtime and Large Functions support are currently beta
  features, so Colab or local Gradio remains the simpler backup demo route.
