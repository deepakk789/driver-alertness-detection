"""
http_routes.py — Standard REST endpoints
=========================================
  GET  /history  →  last N alertness scores (for chart hydration)
  GET  /events   →  full session event log
  POST /reset    →  reset all counters and history
"""

from fastapi import APIRouter
import state

router = APIRouter()


@router.get("/history", summary="Alertness score history")
def get_history():
    """Return the last N alertness data points for the live chart."""
    return {"history": list(state.score_history)}


@router.get("/events", summary="Session event log")
def get_events():
    """Return every distraction / drowsiness event in the current session."""
    return {"events": state.event_log}


@router.post("/reset", summary="Reset session")
def reset_session():
    """Clear all counters, score history, and event log."""
    state.reset()
    return {"status": "Session reset successfully."}
