"""
ws_routes.py — WebSocket endpoint for real-time frame inference
===============================================================
Protocol:
  Client  →  Server :  JSON  { "frame": "<base64 JPEG>" }
  Server  →  Client :  JSON  { eye_status, yawn_status, head_status,
                               eye_prob, yawn_prob, head_prob,
                               drowsiness_score, alert_level, timestamp }
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from inference import run_inference

router = APIRouter()


@router.websocket("/ws")
async def alertness_websocket(websocket: WebSocket):
    """
    Persistent WebSocket connection for streaming webcam frames.

    The client sends one frame at a time and waits for the inference
    result before sending the next — this naturally rate-limits to
    the server's processing speed (≈ 2 fps on CPU).
    """
    await websocket.accept()
    client_host = websocket.client.host
    print(f"[WS] Client connected: {client_host}")

    try:
        while True:
            # ---- Receive frame from browser ----
            raw  = await websocket.receive_text()
            data = json.loads(raw)
            frame_b64 = data.get("frame", "")

            if not frame_b64:
                await websocket.send_text(json.dumps({"error": "No frame provided"}))
                continue

            # ---- Run inference (sync; fast enough at 2 fps) ----
            result = run_inference(frame_b64)

            # ---- Send result back to browser ----
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {client_host}")
    except json.JSONDecodeError as e:
        print(f"[WS] Bad JSON from client: {e}")
        await websocket.close(code=1003)
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        await websocket.close(code=1011)
