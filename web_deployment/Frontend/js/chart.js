/**
 * chart.js — Chart.js initialisation and update helpers
 * ======================================================
 * Exports:
 *   updateChart(timeLabel, score, alertLevel)  — push a new data point
 *   resetChart()                               — clear the chart
 */

const chartCanvas = document.getElementById("alertnessChart");

export const alertChart = new Chart(chartCanvas, {
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
        min: 0, max: 120,
        ticks: { color: "#4a5166" },
        grid:  { color: "rgba(255,255,255,0.04)" },
      }
    }
  }
});

/** Push one new data point and re-render. */
export function updateChart(timeLabel, score, alertLevel) {
  alertChart.data.labels.push(timeLabel);
  alertChart.data.datasets[0].data.push(score);

  // Keep last 60 points (≈30 s at 2 fps)
  if (alertChart.data.labels.length > 60) {
    alertChart.data.labels.shift();
    alertChart.data.datasets[0].data.shift();
  }

  alertChart.data.datasets[0].borderColor =
    alertLevel === "ALERT"      ? "#22d3a5" :
    alertLevel === "DROWSY"     ? "#ef4444" : "#f59e0b";

  alertChart.update("none");
}

/** Clear all chart data back to blank. */
export function resetChart() {
  alertChart.data.labels = [];
  alertChart.data.datasets[0].data = [];
  alertChart.update();
}
