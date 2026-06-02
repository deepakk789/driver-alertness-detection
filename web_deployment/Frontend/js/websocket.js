/**
 * websocket.js — WebSocket connection lifecycle manager
 * ======================================================
 * Handles: open, close, send, receive, auto-reconnect.
 *
 * Exports:
 *   connect(onResult, onStatus)  — open WS and register callbacks
 *   disconnect()                 — gracefully close WS
 *   sendFrame(base64)            — send a frame for inference
 *   isConnected()                — true if WS is currently OPEN
 *
 * Protocol:
 *   Send    →  JSON { "frame": "<base64 JPEG>" }
 *   Receive →  JSON { eye_status, yawn_status, head_status,
 *                     eye_prob, yawn_prob, head_prob,
 *                     drowsiness_score, alert_level, timestamp }
 */

// Build the WS URL from the current page's host so it works
// both in local dev (ws://localhost:8000/ws) and on Render/production.
const WS_URL = `ws://${window.location.host}/ws`;
const RECONNECT_DELAY_MS = 3000;

let socket          = null;
let _onResult       = null;   // callback(data)  — called per inference result
let _onStatus       = null;   // callback(bool)  — called on connect / disconnect
let _shouldReconnect = false;

// ============================================================
// Public API
// ============================================================

/**
 * Open the WebSocket and register event callbacks.
 * Will automatically try to reconnect if the connection drops.
 *
 * @param {function} onResult  — called with parsed JSON result from server
 * @param {function} onStatus  — called with (connected: boolean)
 */
export function connect(onResult, onStatus) {
  _onResult        = onResult;
  _onStatus        = onStatus;
  _shouldReconnect = true;
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
// Private helpers
// ============================================================

function _open() {
  console.log(`[WS] Connecting → ${WS_URL}`);
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    console.log("[WS] Connected ✓");
    _onStatus?.(true);
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
    _onStatus?.(false);
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
