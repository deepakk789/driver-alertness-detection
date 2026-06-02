/**
 * main.js — Entry point: wires camera, WebSocket, UI, and chart together
 * ========================================================================
 * Flow:
 *   Start  → open camera → _sendNext() → [WS] → onInferenceResult() → _sendNext() → ...
 *   Stop   → close camera, clear timer
 *   Reset  → stop + reset server state + clear all UI
 */

import { connect, sendFrame }           from "./websocket.js";
import { startCamera, stopCamera, captureFrame } from "./camera.js";
import {
  setStatus, setServerStatus,
  updateBadges, updateScoreBar,
  logEvent, resetBadges,
  overlay, btnStart, btnStop, btnReset,
  sessionTimerEl, eventLogEl,
  statDuration, statDistract, statDrowsy, statAvgScore,
} from "./ui.js";
import { updateChart, resetChart } from "./chart.js";

// ---- Alarm ----
const alarmSound = new Audio("/static/alarm.wav");
alarmSound.loop  = true;

// ---- Session state ----
let sessionStart         = null;
let sessionTimerInterval = null;
let detecting            = false;
let totalScores          = [];
let distractCount        = 0;
let drowsyCount          = 0;
let lastAlertLevel       = null;

const drowsinessScoreEl = document.getElementById("drowsinessScore");

// ============================================================
// Open WebSocket on page load
// Server-status indicator updates on every connect/disconnect.
// ============================================================
connect(onInferenceResult, (connected) => setServerStatus(connected));

// ============================================================
// Button: Start
// ============================================================
btnStart.addEventListener("click", async () => {
  try {
    await startCamera();
    overlay.classList.add("hidden");
    startSession();
  } catch {
    alert("Camera access denied or not available. Please allow camera permissions.");
  }
});

// ============================================================
// Button: Stop
// ============================================================
btnStop.addEventListener("click", stopSession);

// ============================================================
// Button: Reset
// ============================================================
btnReset.addEventListener("click", async () => {
  stopSession();
  sessionStart = null;

  await fetch("/reset", { method: "POST" }).catch(() => {});

  totalScores   = [];
  distractCount = drowsyCount = 0;
  lastAlertLevel = null;

  statDistract.textContent      = "0";
  statDrowsy.textContent        = "0";
  statAvgScore.textContent      = "0";
  statDuration.textContent      = "0:00";
  sessionTimerEl.textContent    = "Session: 00:00";
  drowsinessScoreEl.textContent = "0";

  resetChart();
  resetBadges();
  setStatus("⏳", "Click Start to begin", "", "");
  overlay.classList.remove("hidden");
  eventLogEl.innerHTML = '<div class="event-placeholder">Session reset. Click Start to begin again.</div>';

  alarmSound.pause();
  alarmSound.currentTime = 0;
});

// ============================================================
// Session lifecycle
// ============================================================

function startSession() {
  detecting      = true;
  sessionStart   = Date.now();
  totalScores    = [];
  distractCount  = drowsyCount = 0;
  lastAlertLevel = null;

  sessionTimerInterval = setInterval(updateTimer, 1000);

  // Kick off the first frame
  sendNext();
}

function stopSession() {
  if (!detecting) return;
  detecting = false;
  clearInterval(sessionTimerInterval);
  stopCamera();
  setStatus("🛑", "Detection Stopped", "Camera released", "");
  overlay.classList.remove("hidden");
  alarmSound.pause();
  alarmSound.currentTime = 0;
}

function updateTimer() {
  if (!sessionStart) return;
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const s = String(elapsed % 60).padStart(2, "0");
  sessionTimerEl.textContent = `Session: ${m}:${s}`;
  statDuration.textContent   = `${m}:${s}`;
}

// ============================================================
// Detection loop — request / response cadence
// One frame is sent; the next is sent only AFTER the result arrives.
// This naturally rate-limits to the server's inference speed.
// ============================================================

function sendNext() {
  if (!detecting) return;

  const frame = captureFrame();
  if (!frame) {
    // Camera not ready yet — retry shortly
    setTimeout(sendNext, 200);
    return;
  }

  const sent = sendFrame(frame);
  if (!sent) {
    // WebSocket reconnecting — retry shortly
    setTimeout(sendNext, 500);
  }
  // On success: wait for onInferenceResult to call sendNext again
}

// ============================================================
// Handle inference results from server
// ============================================================

function onInferenceResult(data) {
  if (data.error) {
    console.warn("[Inference] Server error:", data.error);
    if (detecting) setTimeout(sendNext, 300);
    return;
  }

  const { alert_level, drowsiness_score, timestamp } = data;
  const timeLabel = timestamp
    ? timestamp.slice(11, 19)
    : new Date().toLocaleTimeString();

  // ---- Status card ----
  const statusMap = {
    ALERT:      { icon: "✅", sub: "Driver is alert and attentive",   cls: "alert"      },
    DISTRACTED: { icon: "⚠️",  sub: "Focus lost or camera blocked",    cls: "distracted" },
    DROWSY:     { icon: "🚨", sub: "Drowsiness detected! Pull over!",  cls: "high"       },
  };
  const { icon, sub, cls } = statusMap[alert_level] ?? { icon: "❓", sub: "", cls: "" };
  setStatus(icon, alert_level, sub, cls);

  // ---- Badges & score ----
  updateBadges(data);
  updateScoreBar(drowsiness_score);

  // ---- Chart ----
  updateChart(timeLabel, drowsiness_score, alert_level);

  // ---- Running stats ----
  totalScores.push(drowsiness_score);
  statAvgScore.textContent = Math.round(
    totalScores.reduce((a, b) => a + b, 0) / totalScores.length
  );

  if (alert_level !== lastAlertLevel && alert_level !== "ALERT") {
    if (alert_level === "DISTRACTED") {
      statDistract.textContent = ++distractCount;
      logEvent(timeLabel, "DISTRACTED", "orange");
    } else if (alert_level === "DROWSY") {
      statDrowsy.textContent = ++drowsyCount;
      logEvent(timeLabel, "DROWSY", "red");
    }
  }
  lastAlertLevel = alert_level;

  // ---- Alarm ----
  if (alert_level === "DISTRACTED" || alert_level === "DROWSY") {
    if (alarmSound.paused) alarmSound.play().catch(() => {});
  } else {
    if (!alarmSound.paused) { alarmSound.pause(); alarmSound.currentTime = 0; }
  }

  // ---- Next frame ----
  if (detecting) sendNext();
}
