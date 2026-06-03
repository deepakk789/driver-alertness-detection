"""
ws_routes.py — WebSocket endpoint for real-time frame inference
===============================================================
Each connecting client gets:
  - A unique session_id (UUID)
  - Its own SessionState instance registered in state.sessions
  - Automatic cleanup on disconnect

Protocol:
  Client  ->  Server :  JSON  { "frame": "<base64 JPEG>" }
  Server  ->  Client :  JSON  { eye_status, yawn_status, head_status,
                                eye_prob, yawn_prob, head_prob,
                                drowsiness_score, alert_level,
                                timestamp, session_id }
"""

import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import state
from state import SessionState
from inference import run_inference

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws")
async def alertness_websocket(websocket: WebSocket):
    """
    One WebSocket connection = one isolated driver session.

    The session_id is echoed back in every response so the frontend
    can use it to query /history?session_id=... or POST /reset.
    """
    await websocket.accept()

    session_id  = str(uuid.uuid4())
    session     = SessionState()
    state.sessions[session_id] = session

    client = websocket.client.host if websocket.client else "unknown"
    logger.info("WebSocket connected  session=%s  client=%s", session_id, client)

    try:
        while True:
            # ---- Receive frame ----
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            frame_b64 = data.get("frame", "")
            if not frame_b64:
                await websocket.send_text(json.dumps({"error": "No frame provided"}))
                continue

            # ---- Run inference against this session's isolated state ----
            result = run_inference(frame_b64, session)
            result["session_id"] = session_id

            # ---- Send result back ----
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected  session=%s  client=%s", session_id, client)
    except Exception as e:
        logger.error("Unexpected WebSocket error  session=%s: %s", session_id, e)
        await websocket.close(code=1011)
    finally:
        # Always clean up the session — prevents memory leaks
        state.sessions.pop(session_id, None)
        logger.info("Session removed  session=%s", session_id)
