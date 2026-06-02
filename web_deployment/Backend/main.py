"""
Driver Alertness Detection System — FastAPI Backend (v2)
=========================================================
This file is intentionally thin:
  - Creates the FastAPI app
  - Adds CORS middleware
  - Mounts static files
  - Registers route modules (HTTP + WebSocket)
  - Serves the frontend dashboard at /

Inference logic   → inference.py
Model loading     → models.py
Session state     → state.py
HTTP routes       → routes/http_routes.py
WebSocket route   → routes/ws_routes.py
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routes.http_routes import router as http_router
from routes.ws_routes   import router as ws_router

# ---- App ----
app = FastAPI(
    title="Driver Alertness Detection API",
    version="2.0.0",
    description="Real-time driver drowsiness and distraction detection via WebSocket.",
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Static files (Frontend) ----
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Frontend"))
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ---- Routes ----
app.include_router(http_router)   # GET /history, GET /events, POST /reset
app.include_router(ws_router)     # WS  /ws


# ---- Dashboard entry point ----
@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the frontend HTML dashboard."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
