/**
 * Driver Alertness System — Frontend Logic
 * =========================================
 * 1. Accesses the webcam using getUserMedia.
 * 2. Captures a frame every 500ms using an off-screen canvas.
 * 3. Sends the frame as a base64 JPEG to POST /predict.
 * 4. Updates the UI, live chart, event log, and session stats.
 */

const API_BASE = ""; // empty = same server (localhost:8000 in dev, Render URL in prod)

// ---- DOM References ----
const webcamEl    = document.getElementById("webcam");
const canvasEl    = document.getElementById("captureCanvas");
const ctx         = canvasEl.getContext("2d");
const overlay     = document.getElementById("cameraOverlay");
const btnStart    = document.getElementById("btnStart");
const btnStop     = document.getElementById("btnStop");
const btnReset    = document.getElementById("btnReset");
const serverStatus= document.getElementById("serverStatus");
const sessionTimerEl = document.getElementById("sessionTimer");

// Status
const statusCard  = document.getElementById("statusCard");
const statusIcon  = document.getElementById("statusIcon");
const statusText  = document.getElementById("statusText");
const statusSub   = document.getElementById("statusSub");

// Badges
const eyeStatus   = document.getElementById("eyeStatus");
const yawnStatus  = document.getElementById("yawnStatus");
const headStatus  = document.getElementById("headStatus");
const eyeProb     = document.getElementById("eyeProb");
const yawnProb    = document.getElementById("yawnProb");
const headProb    = document.getElementById("headProb");
const eyeBar      = document.getElementById("eyeBar");
const yawnBar     = document.getElementById("yawnBar");
const headBar     = document.getElementById("headBar");
const badgeEye    = document.getElementById("badgeEye");
const badgeYawn   = document.getElementById("badgeYawn");
const badgeHead   = document.getElementById("badgeHead");

// Score
const drowsinessScore = document.getElementById("drowsinessScore");
const scoreFill       = document.getElementById("scoreFill");

// Analytics
const statDuration = document.getElementById("statDuration");
const statDistract = document.getElementById("statDistract");
const statDrowsy   = document.getElementById("statDrowsy");
const statAvgScore = document.getElementById("statAvgScore");
const eventLog     = document.getElementById("eventLog");

// ---- State ----
let detecting = false;
let detectInterval = null;
let sessionStart   = null;
let sessionTimerInterval = null;
let totalScores    = [];
let distractCount  = 0;
let drowsyCount    = 0;
let lastAlertLevel = null;

// ---- Alarm Setup ----
const alarmSound = new Audio("/static/alarm.wav");
alarmSound.loop = true;

// ---- Chart Setup ----
const chartCanvas = document.getElementById("alertnessChart");
const alertChart  = new Chart(chartCanvas, {
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label: "Drowsiness Score",
      data: [],
      borderColor: "#22d3a5",
      backgroundColor: "rgba(34,211,165,0.08)",
      borderWidth: 2,
      tension: 0.4,
      pointRadius: 0,
      fill: true,
    }]
  },
  options: {
    responsive: true,
    animation: { duration: 200 },
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { color: "#4a5166", maxTicksLimit: 6, maxRotation: 0 },
        grid:  { color: "rgba(255,255,255,0.04)" },
      },
      y: {
        min: 0,
        max: 120,
        ticks: { color: "#4a5166" },
        grid:  { color: "rgba(255,255,255,0.04)" },
      }
    }
  }
});

// ---- Check Server Status ----
async function pingServer() {
  try {
    const res = await fetch(`${API_BASE}/history`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      serverStatus.className = "server-status online";
      serverStatus.innerHTML = '<span class="dot"></span> Server Online';
    }
  } catch {
    serverStatus.className = "server-status offline";
    serverStatus.innerHTML = '<span class="dot"></span> Server Offline';
  }
}
pingServer();
setInterval(pingServer, 10000);

// ---- Start Camera ----
btnStart.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    webcamEl.srcObject = stream;
    overlay.classList.add("hidden");
    startSession();
  } catch (err) {
    alert("Camera access denied or not available. Please allow camera permissions.");
    console.error(err);
  }
});

// ---- Session ----
function startSession() {
  detecting    = true;
  sessionStart = Date.now();
  distractCount  = 0;
  drowsyCount    = 0;
  totalScores    = [];
  lastAlertLevel = null;

  // Session timer display
  sessionTimerInterval = setInterval(updateSessionTimer, 1000);

  // Start sending frames sequentially
  detectionLoop();
}

async function detectionLoop() {
  if (!detecting) return;
  await captureAndSend();
  // Wait 500ms before sending the next frame
  setTimeout(detectionLoop, 500);
}

function updateSessionTimer() {
  if (!sessionStart) return;
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const m = Math.floor(elapsed / 60).toString().padStart(2, "0");
  const s = (elapsed % 60).toString().padStart(2, "0");
  sessionTimerEl.textContent = `Session: ${m}:${s}`;
  statDuration.textContent   = `${m}:${s}`;
}

// ---- Stop ----
btnStop.addEventListener("click", () => {
  if (!detecting) return;
  
  // Stop detection
  clearInterval(sessionTimerInterval);
  detecting = false;

  // Release camera
  if (webcamEl.srcObject) {
    webcamEl.srcObject.getTracks().forEach(track => track.stop());
    webcamEl.srcObject = null;
  }

  // Update UI
  setStatus("STOPPED", "🛑", "Detection Stopped", "Camera released");
  overlay.classList.remove("hidden");

  // Stop alarm
  alarmSound.pause();
  alarmSound.currentTime = 0;
});

// ---- Reset ----
btnReset.addEventListener("click", async () => {
  // Call the stop logic first to ensure intervals and camera are cleared
  if (detecting) {
    clearInterval(sessionTimerInterval);
    detecting = false;
    if (webcamEl.srcObject) {
      webcamEl.srcObject.getTracks().forEach(track => track.stop());
      webcamEl.srcObject = null;
    }
  }

  sessionStart = null;

  // Reset server counters
  await fetch(`${API_BASE}/reset`, { method: "POST" }).catch(() => {});

  // Reset local state
  distractCount  = 0;
  drowsyCount    = 0;
  totalScores    = [];
  lastAlertLevel = null;

  // Reset UI
  statDistract.textContent = "0";
  statDrowsy.textContent   = "0";
  statAvgScore.textContent = "0";
  statDuration.textContent = "0:00";
  sessionTimerEl.textContent = "Session: 00:00";
  drowsinessScore.textContent = "0";
  scoreFill.style.width = "0%";

  // Clear chart
  alertChart.data.labels = [];
  alertChart.data.datasets[0].data = [];
  alertChart.update();

  // Clear event log
  eventLog.innerHTML = '<div class="event-placeholder">Session reset. Click Start to begin again.</div>';

  // Reset badges
  resetBadges();
  setStatus("WAITING", "⏳", "Click Start to begin", "");

  // Show overlay again
  overlay.classList.remove("hidden");

  // Stop alarm
  alarmSound.pause();
  alarmSound.currentTime = 0;
});

// ---- Capture Frame & Send ----
async function captureAndSend() {
  if (!detecting || !webcamEl.srcObject) return;

  const width  = webcamEl.videoWidth  || 640;
  const height = webcamEl.videoHeight || 480;
  canvasEl.width  = width;
  canvasEl.height = height;

  // Draw mirrored frame (same as what user sees)
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(webcamEl, -width, 0, width, height);
  ctx.restore();

  // Export as JPEG (quality 0.8 = good balance of speed vs quality)
  const base64 = canvasEl.toDataURL("image/jpeg", 0.8).split(",")[1];

  try {
    const res  = await fetch(`${API_BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ frame: base64 }),
      signal:  AbortSignal.timeout(15000) // Increase timeout for slow Render CPUs
    });
    const data = await res.json();
    updateUI(data);
  } catch (err) {
    console.warn("Prediction failed:", err.message);
  }
}

// ---- Update UI ----
function updateUI(data) {
  const {
    eye_status, yawn_status, head_status,
    eye_prob, yawn_prob, head_prob,
    drowsiness_score, alert_level, timestamp
  } = data;

  // -- Status card --
  let icon, sub, cardClass;
  if (alert_level === "ALERT") {
    icon = "✅"; sub = "Driver is alert and attentive"; cardClass = "alert";
  } else if (alert_level === "DISTRACTED") {
    icon = "⚠️"; sub = "Focus lost or camera blocked"; cardClass = "distracted";
  } else if (alert_level === "DROWSY") {
    icon = "🚨"; sub = "Drowsiness detected! STOP!"; cardClass = "high";
  } else {
    icon = "❓"; sub = "Status unknown"; cardClass = "";
  }
  setStatus(alert_level, icon, alert_level, sub, cardClass);

  // -- Badges --
  updateBadge(badgeEye,  eyeStatus,  eyeProb,  eyeBar,  eye_status,  eye_prob,
    eye_status === "CLOSED" ? "danger" : "safe");
  updateBadge(badgeYawn, yawnStatus, yawnProb, yawnBar, yawn_status, yawn_prob,
    yawn_status === "YAWN" ? "danger" : "safe");
  updateBadge(badgeHead, headStatus, headProb, headBar, head_status, head_prob,
    head_status === "AWAY" ? "warning" : "safe");

  // -- Score bar (max cap 120) --
  const capped = Math.min(drowsiness_score, 120);
  drowsinessScore.textContent = drowsiness_score;
  scoreFill.style.width = `${(capped / 120) * 100}%`;

  // -- Chart --
  const timeLabel = timestamp ? timestamp.slice(11, 19) : new Date().toLocaleTimeString();
  alertChart.data.labels.push(timeLabel);
  alertChart.data.datasets[0].data.push(drowsiness_score);
  if (alertChart.data.labels.length > 60) {
    alertChart.data.labels.shift();
    alertChart.data.datasets[0].data.shift();
  }

  // Update chart line color based on alert level
  const lineColor = alert_level === "ALERT" ? "#22d3a5" :
                    alert_level === "MILD DROWSINESS" ? "#f59e0b" : "#ef4444";
  alertChart.data.datasets[0].borderColor = lineColor;
  alertChart.update("none");

  // -- Track events --
  totalScores.push(drowsiness_score);
  statAvgScore.textContent = Math.round(totalScores.reduce((a,b)=>a+b,0)/totalScores.length);

  if (alert_level !== lastAlertLevel && alert_level !== "ALERT") {
    if (alert_level === "DISTRACTED") {
      distractCount++;
      statDistract.textContent = distractCount;
      logEvent(timeLabel, alert_level, "orange");
    } else if (alert_level === "DROWSY") {
      drowsyCount++;
      statDrowsy.textContent = drowsyCount;
      logEvent(timeLabel, alert_level, "red");
    }
  }
  lastAlertLevel = alert_level;

  // -- Alarm Logic --
  if (alert_level === "DISTRACTED" || alert_level === "DROWSY") {
    if (alarmSound.paused) {
      alarmSound.play().catch(e => console.warn("Audio play blocked by browser:", e));
    }
  } else {
    if (!alarmSound.paused) {
      alarmSound.pause();
      alarmSound.currentTime = 0;
    }
  }
}

// ---- Helpers ----
function setStatus(text, icon, mainText, sub, cardClass) {
  statusIcon.textContent = icon;
  statusText.textContent = mainText;
  statusSub.textContent  = sub;
  statusCard.className   = `status-card ${cardClass || ""}`;
}

function updateBadge(badgeEl, valEl, probEl, barEl, status, prob, cls) {
  valEl.textContent  = status;
  probEl.textContent = `${(prob * 100).toFixed(0)}%`;
  barEl.style.width  = `${prob * 100}%`;
  barEl.style.background = cls === "danger" ? "#ef4444" : cls === "warning" ? "#f59e0b" : "#22d3a5";
  badgeEl.className  = `badge ${cls}`;
}

function resetBadges() {
  ["eyeStatus","yawnStatus","headStatus"].forEach(id => document.getElementById(id).textContent = "—");
  ["eyeProb","yawnProb","headProb"].forEach(id => document.getElementById(id).textContent = "—");
  ["eyeBar","yawnBar","headBar"].forEach(id => {
    document.getElementById(id).style.width = "0%";
  });
  ["badgeEye","badgeYawn","badgeHead"].forEach(id => {
    document.getElementById(id).className = "badge";
  });
}

function logEvent(time, type, color) {
  const placeholder = eventLog.querySelector(".event-placeholder");
  if (placeholder) placeholder.remove();

  const item = document.createElement("div");
  item.className = "event-item";
  item.innerHTML = `
    <div class="event-dot ${color}"></div>
    <div class="event-time">${time}</div>
    <div class="event-type">${type}</div>
  `;
  // Prepend so newest is at top
  eventLog.insertBefore(item, eventLog.firstChild);

  // Keep max 50 events
  while (eventLog.children.length > 50) {
    eventLog.removeChild(eventLog.lastChild);
  }
}
