# Deploying to Vercel and Using with a Frontend

Yes — you can deploy this FastAPI backend to **Vercel** and call it from any frontend (React, Next.js, Vue, etc.).

## 1. Deploy the API to Vercel

- **From Git:** Push this repo and [import the project in Vercel](https://vercel.com/new). Vercel will detect the FastAPI app from `index.py`.
- **From CLI:** Install [Vercel CLI](https://vercel.com/docs/cli), then from the project root run:
  ```bash
  vercel
  ```

## 2. Set environment variables

In the Vercel project: **Settings → Environment Variables**, add:

- `GROQ_API_KEY`
- `REPLICATE_API_TOKEN` (if using Replicate)
- `LEONARDO_API_KEY` (if using Leonardo)

Optional: `VIDEO_POLL_INTERVAL`, `VIDEO_MAX_WAIT` (see `.env.example`).

Redeploy after changing env vars.

## 3. Use the API from your frontend

After deploy, your API base URL will be like:

`https://your-project.vercel.app`

Use it in the frontend as the base for all API calls, for example:

- `POST https://your-project.vercel.app/api/extract-options`
- `POST https://your-project.vercel.app/api/enhance-prompt`
- `POST https://your-project.vercel.app/api/generate-script`
- `POST https://your-project.vercel.app/api/generate-video`
- `GET https://your-project.vercel.app/api/health`

CORS is already configured to allow any origin (`allow_origins=["*"]`), so browser-based frontends can call the API without extra CORS setup.

## 4. Video generation and timeouts

The **generate-video** endpoint can run for a long time (it polls Leonardo/Replicate until the video is ready, up to `VIDEO_MAX_WAIT` seconds, default 900).

- **Vercel limits:** On the Pro plan, the maximum function duration is **800 seconds** (~13 minutes). The default in `vercel.json` is **300 seconds** (5 minutes).
- If video jobs often take longer than 5 minutes, either:
  - Increase `maxDuration` in `vercel.json` (e.g. to `800` on Pro), and/or
  - Lower `VIDEO_MAX_WAIT` in env vars so the backend stops polling earlier and returns a timeout error instead of hanging.

For very long or heavy video workloads, consider running the API on a platform with higher or no request timeout (e.g. Railway, Render, Fly.io) and keep the frontend on Vercel.

## 5. Optional: frontend on the same Vercel project

You can host a frontend in the same repo:

- Put the frontend app in a subfolder (e.g. `frontend/`) and set the Vercel **Root Directory** to that folder, **or**
- Use a monorepo with the backend at the root and the frontend in a subfolder, and configure [Vercel for monorepos](https://vercel.com/docs/monorepos) (e.g. one project for the API, one for the frontend, or a single project with rewrites).

Then point the frontend’s API base URL to your deployed backend URL (e.g. `https://your-api.vercel.app`).
