"""
Vercel serverless function: expose the FastAPI app (must live in api/ for functions config).
"""
import sys
from pathlib import Path

# Ensure project root is on path when Vercel runs from api/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from main import app
