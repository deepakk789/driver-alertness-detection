"""
http_routes.py — Standard REST endpoints
=========================================
All endpoints accept an optional ?session_id= query parameter.
If provided, they operate on that specific session's data.
If omitted and only one session is active, they fall back to it.

  GET  /history              -> last N alertness scores
  GET  /events               -> session event log
  POST /reset                -> reset counters for a session
  GET  /health               -> server health check
  GET  /sessions             -> list of active session IDs
"""

import logging
from fastapi import APIRouter, Query, HTTPException

import state
from models import models_ready

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_session(session_id: str | None):
    """
    Look up a session by ID. If no ID is given and exactly one
    session is active, return that one. Otherwise raise a 404.
    """
    if session_id:
        s = state.sessions.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return s

    # No session_id given — fall back if only one is active
    if len(state.sessions) == 1:
        return next(iter(state.sessions.values()))

    if len(state.sessions) == 0:
        raise HTTPException(status_code=404, detail="No active sessions.")

    raise HTTPException(
        status_code=400,
        detail="Multiple sessions active. Provide ?session_id=<id>."
    )


@router.get("/history", summary="Alertness score history")
def get_history(session_id: str | None = Query(default=None)):
    """Return the last N alertness data points for the live chart."""
    session = _get_session(session_id)
    return {"history": list(session.score_history)}


@router.get("/events", summary="Session event log")
def get_events(session_id: str | None = Query(default=None)):
    """Return all distraction and drowsiness events in the session."""
    session = _get_session(session_id)
    return {"events": session.event_log}


@router.post("/reset", summary="Reset session")
def reset_session(session_id: str | None = Query(default=None)):
    """Clear all counters, score history, and event log for a session."""
    session = _get_session(session_id)
    session.reset()
    logger.info("Session reset  session_id=%s", session_id)
    return {"status": "Session reset successfully."}


@router.get("/health", summary="Health check")
def health():
    """Liveness probe — confirms server is up and whether models are loaded."""
    return {
        "status":           "ok",
        "models_loaded":    models_ready.is_set(),
        "active_sessions":  len(state.sessions),
    }


@router.get("/sessions", summary="List active sessions")
def list_sessions():
    """Return all currently active session IDs."""
    return {"active_sessions": list(state.sessions.keys())}
