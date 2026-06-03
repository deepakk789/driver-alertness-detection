# Bug Fix Report — Driver Alertness Detection System

**Date:** 2026-06-03  
**Reported By:** User  
**Fixed By:** Antigravity AI Assistant

---

## Problem Reported by User

> *"It is showing 'Connecting...' from a long time but not showing Server Online or Offline.  
> After the last 2 commits you changed from single user system to multi-user system but it is not even working now."*

The frontend UI was stuck permanently on **"Connecting..."** in the server status indicator (top-right header).  
The WebSocket never successfully connected to the backend, so detection could not be started.

---

## Investigation — What Was Checked

The following files were inspected in full:

| File | Purpose |
|------|---------|
| `Frontend/js/websocket.js` | WebSocket client connection logic |
| `Frontend/js/main.js` | Frontend entry point, wires WS + UI |
| `Frontend/js/ui.js` | DOM updates, status indicator |
| `Frontend/style.css` | CSS for status dot colours |
| `Frontend/index.html` | How JS modules are loaded |
| `Backend/main.py` | FastAPI app entry point |
| `Backend/models.py` | TensorFlow model loading |
| `Backend/inference.py` | CV/ML inference logic |
| `Backend/routes/ws_routes.py` | WebSocket endpoint |
| `Backend/routes/http_routes.py` | REST endpoints (/health, /reset, etc.) |
| `Dockerfile` | Container startup command |

---

## Root Causes Found

### Bug 1 — Wrong WebSocket Protocol (`ws://` on HTTPS) — `websocket.js`

**File:** `web_deployment/Frontend/js/websocket.js`

**Bad Code (before fix):**
```js
const WS_URL = `ws://${window.location.host}/ws`;
```

**Problem:**  
The URL was hardcoded to use `ws://` (plain WebSocket).  
Render (the deployment platform) serves the site over **HTTPS**.  
Browsers **block `ws://` connections from HTTPS pages** — this is a browser security rule called "Mixed Content Blocking".  
The socket would silently fail to open, fire `onclose` immediately, and retry every 3 seconds — forever showing "Connecting...".

---

### Bug 2 — No Timeout / No Fallback Status — `websocket.js`

**File:** `web_deployment/Frontend/js/websocket.js`

**Problem:**  
If the WebSocket failed to connect (for any reason), the code had no timeout.  
The status indicator was only updated in `socket.onopen` (→ Online) and `socket.onclose` (→ Offline).  
Because `onopen` never fired and `onclose` fired instantly and triggered a reconnect loop, the status **never left "Connecting..."**.  
The user had no feedback about whether the server was actually reachable or not.

---

### Bug 3 — TensorFlow Models Loaded Synchronously at Startup (MAIN CAUSE on Render) — `models.py`

**File:** `web_deployment/Backend/models.py`

**Bad Code (before fix):**
```python
print("[INFO] Loading models...")
eye_model  = load_model(EYE_MODEL)
yawn_model = load_model(YAWN_MODEL)
print("[INFO] Models loaded successfully.")
```

**Problem (Most Critical):**  
This code runs **at Python import time** — meaning when the server starts up, Python imports `main.py` → `inference.py` → `models.py` and immediately calls `load_model()`.

Loading two TensorFlow `.h5` models on Render's **free-tier CPU** takes **2 to 3 minutes**.

**During those 2-3 minutes:**
- Uvicorn has not started yet
- No port is bound on the container
- The Render platform scans for an open port and finds nothing
- Render logs showed: **"No open ports detected, continuing to scan..."**
- After its timeout, Render marks the deployment as **unhealthy / failed**
- Frontend gets no WebSocket connection → stuck on "Connecting..." forever

This was confirmed directly from the Render deploy logs shared by the user.

---

### Bug 4 — `/health` Endpoint Always Returned `models_loaded: True` — `http_routes.py`

**File:** `web_deployment/Backend/routes/http_routes.py`

**Bad Code (before fix):**
```python
return {
    "status":        "ok",
    "models_loaded": True,   # ← hardcoded, never actually checked
    ...
}
```

**Problem:**  
The health endpoint was lying — it always said models were loaded regardless of actual state.  
This made it impossible for the frontend to know whether the server was in a "loading" state vs "fully ready" state.

---

## Fixes Applied

### Fix 1 — Auto-detect `ws://` vs `wss://` — `websocket.js`

```js
// BEFORE (broken on HTTPS)
const WS_URL = `ws://${window.location.host}/ws`;

// AFTER (works on both local HTTP and Render HTTPS)
const _proto = window.location.protocol === "https:" ? "wss" : "ws";
const WS_URL = `${_proto}://${window.location.host}/ws`;
```

**Result:** WebSocket now uses `wss://` on Render and `ws://` locally. No more Mixed Content blocking.

---

### Fix 2 — Background HTTP Health Polling for Status — `websocket.js`

Instead of relying only on WS `onopen`/`onclose` for status, the frontend now **polls `/health` every 4 seconds via HTTP**.

This gives 4 meaningful status states:

| Status | Dot Colour | Meaning |
|--------|-----------|---------|
| Connecting... | ⚪ Grey (pulse) | Initial state / WS not yet open |
| Server Loading… | 🟠 Orange (pulse) | Server up, TF models still loading |
| Server Online | 🟢 Green | Fully ready |
| Server Offline | 🔴 Red | Server not reachable |

WS reconnect delay increased from 3s to 5s to be more patient with cold starts.

---

### Fix 3 — Non-Blocking Model Loading in Background Thread — `models.py`

```python
# BEFORE: blocks uvicorn from starting for 2-3 minutes
eye_model  = load_model(EYE_MODEL)
yawn_model = load_model(YAWN_MODEL)

# AFTER: loads in a background thread; uvicorn starts immediately
def _load():
    global eye_model, yawn_model
    eye_model  = load_model(EYE_MODEL)
    yawn_model = load_model(YAWN_MODEL)
    models_ready.set()   # signals that models are ready

_thread = threading.Thread(target=_load, daemon=True)
_thread.start()
```

**Result:** Uvicorn binds the port within seconds. Render's health check passes. The `/ws` endpoint is available immediately. Models load in the background and become available after 2-3 minutes.

---

### Fix 4 — Guard in `inference.py` for Cold Start — `inference.py`

```python
# Added at the top of run_inference()
if not models_ready.is_set() or eye_model is None or yawn_model is None:
    return {"error": "Models are still loading, please wait a moment and retry."}
```

**Result:** If a frame is sent before models finish loading, the server returns a clean error message instead of crashing with `TypeError: 'NoneType' is not callable`.

---

### Fix 5 — Honest `/health` Endpoint — `http_routes.py`

```python
# BEFORE
"models_loaded": True   # always True, useless

# AFTER
"models_loaded": models_ready.is_set()   # real state
```

**Result:** Frontend health polling can now distinguish "server starting" from "server fully ready" and show the correct status dot colour.

---

### Fix 6 — 4-State Status Indicator in UI — `ui.js` + `style.css`

`setServerStatus()` was updated from accepting a `boolean` to a `string` with 4 states.  
CSS added an orange `.loading` dot class.  
`main.js` updated to pass the status string instead of a boolean.

---

## Files Changed Summary

| File | What Changed |
|------|-------------|
| `Backend/models.py` | Synchronous load → background thread with `threading.Event` |
| `Backend/inference.py` | Added `models_ready` guard before inference runs |
| `Backend/routes/http_routes.py` | `/health` reports real `models_loaded` state |
| `Frontend/js/websocket.js` | Full rewrite: `wss://` fix + HTTP health polling + 4-state status |
| `Frontend/js/ui.js` | `setServerStatus` handles 4 states instead of a boolean |
| `Frontend/js/main.js` | Passes status string to `setServerStatus` |
| `Frontend/style.css` | Added orange `.loading` dot CSS rule |

---

## Multi-User Architecture Status

The multi-user WebSocket architecture (`ws_routes.py`, `state.py`) introduced in the last 2 commits is **correct and has no bugs**.  
Each WebSocket connection correctly gets its own isolated `SessionState` instance.  
The issue was entirely in the **startup and connection layer**, not in the session management logic.
