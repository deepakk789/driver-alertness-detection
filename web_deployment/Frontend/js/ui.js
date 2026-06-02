/**
 * ui.js — All DOM reads and writes
 * ==================================
 * Exports DOM element references used by main.js,
 * plus helper functions that update the UI.
 */

// ---- Exported DOM references ----
export const webcamEl       = document.getElementById("webcam");
export const canvasEl       = document.getElementById("captureCanvas");
export const overlay        = document.getElementById("cameraOverlay");
export const btnStart       = document.getElementById("btnStart");
export const btnStop        = document.getElementById("btnStop");
export const btnReset       = document.getElementById("btnReset");
export const serverStatus   = document.getElementById("serverStatus");
export const sessionTimerEl = document.getElementById("sessionTimer");
export const statDuration   = document.getElementById("statDuration");
export const statDistract   = document.getElementById("statDistract");
export const statDrowsy     = document.getElementById("statDrowsy");
export const statAvgScore   = document.getElementById("statAvgScore");
export const eventLogEl     = document.getElementById("eventLog");

// ---- Private refs ----
const statusCard  = document.getElementById("statusCard");
const statusIcon  = document.getElementById("statusIcon");
const statusText  = document.getElementById("statusText");
const statusSub   = document.getElementById("statusSub");
const drowsinessScore = document.getElementById("drowsinessScore");
const scoreFill       = document.getElementById("scoreFill");

// ============================================================
// Status Card
// ============================================================

/**
 * Update the main alertness status card.
 * @param {string} icon      — emoji
 * @param {string} mainText  — heading text
 * @param {string} sub       — sub-text
 * @param {string} cardClass — CSS modifier class
 */
export function setStatus(icon, mainText, sub, cardClass = "") {
  statusIcon.textContent = icon;
  statusText.textContent = mainText;
  statusSub.textContent  = sub;
  statusCard.className   = `status-card ${cardClass}`;
}

// ============================================================
// Detection Badges
// ============================================================

function _updateBadge(badgeEl, valEl, probEl, barEl, status, prob, cls) {
  valEl.textContent  = status;
  probEl.textContent = `${(prob * 100).toFixed(0)}%`;
  barEl.style.width  = `${prob * 100}%`;
  barEl.style.background =
    cls === "danger"  ? "#ef4444" :
    cls === "warning" ? "#f59e0b" : "#22d3a5";
  badgeEl.className = `badge ${cls}`;
}

/** Update all three detection badges from a result object. */
export function updateBadges({ eye_status, eye_prob, yawn_status, yawn_prob, head_status, head_prob }) {
  _updateBadge(
    document.getElementById("badgeEye"),
    document.getElementById("eyeStatus"),
    document.getElementById("eyeProb"),
    document.getElementById("eyeBar"),
    eye_status, eye_prob,
    eye_status === "CLOSED" ? "danger" : "safe"
  );
  _updateBadge(
    document.getElementById("badgeYawn"),
    document.getElementById("yawnStatus"),
    document.getElementById("yawnProb"),
    document.getElementById("yawnBar"),
    yawn_status, yawn_prob,
    yawn_status === "YAWN" ? "danger" : "safe"
  );
  _updateBadge(
    document.getElementById("badgeHead"),
    document.getElementById("headStatus"),
    document.getElementById("headProb"),
    document.getElementById("headBar"),
    head_status, head_prob,
    head_status === "AWAY" ? "warning" : "safe"
  );
}

// ============================================================
// Score Bar
// ============================================================

/** Update the drowsiness score number and progress bar. */
export function updateScoreBar(score) {
  drowsinessScore.textContent = score;
  scoreFill.style.width = `${(Math.min(score, 120) / 120) * 100}%`;
}

// ============================================================
// Event Log
// ============================================================

/** Append a new event row to the event log panel. */
export function logEvent(time, type, color) {
  const placeholder = eventLogEl.querySelector(".event-placeholder");
  if (placeholder) placeholder.remove();

  const item = document.createElement("div");
  item.className = "event-item";
  item.innerHTML = `
    <div class="event-dot ${color}"></div>
    <div class="event-time">${time}</div>
    <div class="event-type">${type}</div>
  `;
  eventLogEl.insertBefore(item, eventLogEl.firstChild);
  while (eventLogEl.children.length > 50) eventLogEl.removeChild(eventLogEl.lastChild);
}

// ============================================================
// Resets
// ============================================================

/** Reset all badge values to placeholder dashes. */
export function resetBadges() {
  ["eyeStatus","yawnStatus","headStatus"].forEach(id =>
    (document.getElementById(id).textContent = "—"));
  ["eyeProb","yawnProb","headProb"].forEach(id =>
    (document.getElementById(id).textContent = "—"));
  ["eyeBar","yawnBar","headBar"].forEach(id =>
    (document.getElementById(id).style.width = "0%"));
  ["badgeEye","badgeYawn","badgeHead"].forEach(id =>
    (document.getElementById(id).className = "badge"));
}

// ============================================================
// Server Status Indicator
// ============================================================

/** Toggle the server status dot between online / offline. */
export function setServerStatus(online) {
  serverStatus.className = `server-status ${online ? "online" : "offline"}`;
  serverStatus.innerHTML = `<span class="dot"></span> Server ${online ? "Online" : "Offline"}`;
}
