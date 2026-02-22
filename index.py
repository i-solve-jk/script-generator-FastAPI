"""
Vercel entrypoint: expose the FastAPI app so Vercel can serve it.
See: https://vercel.com/docs/frameworks/backend/fastapi
"""
from main import app
