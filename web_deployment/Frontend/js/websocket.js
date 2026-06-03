/**
 * websocket.js — WebSocket connection lifecycle manager
 * ======================================================
 * Handles: open, close, send, receive, auto-reconnect.
 *
 * Status is driven by HTTP /health polling (not just WS open/close)
 * so Render cold starts (server up but models loading) are shown
 * correctly as "Loading..." instead of "Offline".
 *
 * Exports:
 *   connect(onResult, onStatus)  — open WS and register callbacks
 *   disconnect()                 — gracefully close WS
 *   sendFrame(base64)            — send a frame for inference
 *   isConnected()                — true if WS is currently OPEN
 *
 * onStatus is called with one of: "connecting" | "loading" | "online" | "offline"
 */

// Auto-detect ws:// vs wss:// based on page protocol.
// Render serves HTTPS, so we need wss:// there.
const _proto = window.location.protocol === "https:" ? "wss" : "ws";
const WS_URL = `${_proto}://${window.location.host}/ws`;

const RECONNECT_DELAY_MS = 5000;   // wait 5 s between WS reconnect attempts
const HEALTH_POLL_MS     = 4000;   // poll /health every 4 s

let socket           = null;
let _onResult        = null;   // callback(data)  — called per inference result
let _onStatus        = null;   // callback(string) — "connecting"|"loading"|"online"|"offline"
let _shouldReconnect = false;
let _healthTimer     = null;
let _wsConnected     = false;

// ============================================================
// Public API
// ============================================================

/**
 * Open the WebSocket and register event callbacks.
 * Starts a /health poll loop so the status indicator is accurate
 * even during Render cold-start model loading.
 *
 * @param {function} onResult  — called with parsed JSON result from server
 * @param {function} onStatus  — called with status string
 */
export function connect(onResult, onStatus) {
  _onResult        = onResult;
  _onStatus        = onStatus;
  _shouldReconnect = true;

  _onStatus?.("connecting");

  // Start health polling — this drives the status dot
  _startHealthPoll();

  // Open the WS in parallel
  _open();
}

/**
 * Send one base64 JPEG frame to the server for inference.
 * @param {string} base64Frame
 * @returns {boolean}  true if sent, false if socket is not ready
 */
export function sendFrame(base64Frame) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify({ frame: base64Frame }));
  return true;
}

/** Gracefully close the WebSocket (no auto-reconnect after this). */
export function disconnect() {
  _shouldReconnect = false;
  _stopHealthPoll();
  if (socket) {
    socket.close(1000, "Session ended by user");
    socket = null;
  }
}

/** Returns true when the WebSocket is in the OPEN state. */
export function isConnected() {
  return socket?.readyState === WebSocket.OPEN;
}

// ============================================================
// Health polling — drives the status indicator
// ============================================================

function _startHealthPoll() {
  _stopHealthPoll();
  _pollHealth();                                    // immediate first check
  _healthTimer = setInterval(_pollHealth, HEALTH_POLL_MS);
}

function _stopHealthPoll() {
  if (_healthTimer !== null) {
    clearInterval(_healthTimer);
    _healthTimer = null;
  }
}

async function _pollHealth() {
  try {
    const res  = await fetch("/health", { signal: AbortSignal.timeout(3000) });
    const data = await res.json();

    if (!res.ok) {
      _onStatus?.("offline");
      return;
    }

    if (!data.models_loaded) {
      // Server is up but TF models are still loading (Render cold start)
      _onStatus?.("loading");
    } else if (_wsConnected) {
      _onStatus?.("online");
    } else {
      // Models ready but WS hasn't opened yet — still connecting
      _onStatus?.("connecting");
    }
  } catch {
    // fetch failed — server not reachable
    _onStatus?.("offline");
  }
}

// ============================================================
// WebSocket lifecycle
// ============================================================

function _open() {
  if (!_shouldReconnect) return;
  console.log(`[WS] Connecting → ${WS_URL}`);
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    console.log("[WS] Connected ✓");
    _wsConnected = true;
    // Health poll will pick this up on next tick and call _onStatus("online")
    _pollHealth();
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      _onResult?.(data);
    } catch (e) {
      console.warn("[WS] Could not parse message:", e);
    }
  };

  socket.onclose = (event) => {
    console.warn(`[WS] Disconnected (code ${event.code})`);
    _wsConnected = false;
    if (_shouldReconnect) {
      console.log(`[WS] Reconnecting in ${RECONNECT_DELAY_MS / 1000}s…`);
      setTimeout(_open, RECONNECT_DELAY_MS);
    }
  };

  socket.onerror = (err) => {
    // onerror always fires just before onclose — logging is enough here
    console.error("[WS] Socket error:", err);
  };
}
