/**
 * camera.js — Webcam access and frame capture
 * ============================================
 * Exports:
 *   startCamera()   → Promise<MediaStream>  — requests camera, starts <video>
 *   stopCamera()                            — releases all camera tracks
 *   captureFrame()  → string|null           — base64 JPEG of current frame
 */

import { webcamEl, canvasEl } from "./ui.js";

const ctx = canvasEl.getContext("2d");

/**
 * Request webcam access and pipe the stream into the <video> element.
 * Throws if the user denies permission.
 */
export async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 }
  });
  webcamEl.srcObject = stream;
  return stream;
}

/** Stop all camera tracks and clear the <video> source. */
export function stopCamera() {
  if (webcamEl.srcObject) {
    webcamEl.srcObject.getTracks().forEach(t => t.stop());
    webcamEl.srcObject = null;
  }
}

/**
 * Draw the current video frame (mirrored) onto the hidden canvas
 * and return it as a base64 JPEG string (no "data:..." prefix).
 * Returns null if no camera is active.
 */
export function captureFrame() {
  if (!webcamEl.srcObject) return null;

  const w = webcamEl.videoWidth  || 640;
  const h = webcamEl.videoHeight || 480;
  canvasEl.width  = w;
  canvasEl.height = h;

  // Mirror so the frame matches what the user sees on screen
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(webcamEl, -w, 0, w, h);
  ctx.restore();

  // quality 0.8 = good balance of speed vs accuracy
  return canvasEl.toDataURL("image/jpeg", 0.8).split(",")[1];
}
