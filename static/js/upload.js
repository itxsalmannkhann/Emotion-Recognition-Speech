/**
 * VoiceIQ — Upload Page Script
 * Handles: drag & drop, file validation, API call, results rendering
 */

const dropZone = document.getElementById('dropZone');
const dropZoneInner = document.getElementById('dropZoneInner');
const filePreview = document.getElementById('filePreview');
const audioInput = document.getElementById('audioInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const audioPlayer = document.getElementById('audioPlayer');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');

let currentFile = null;
let lastResult = null;
let emotionChart = null;
let waveformChart = null;

const LOADING_STEPS = [
  'Extracting acoustic features...',
  'Computing MFCC & Chroma features...',
  'Running LSTM neural network...',
  'Estimating gender and age...',
  'Finalizing predictions...',
];

// ─── DROP ZONE EVENTS ────────────────────────

dropZone.addEventListener('click', (e) => {
  if (e.target !== dropZoneInner && !dropZoneInner.contains(e.target)) return;
  audioInput.click();
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

audioInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) setFile(file);
});

document.getElementById('removeFile')?.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
});


// ─── FILE HANDLING ────────────────────────────

const ALLOWED_TYPES = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/flac', 'audio/x-flac', 'audio/mp4', 'audio/m4a'];
const ALLOWED_EXTS = ['wav', 'mp3', 'ogg', 'flac', 'm4a'];

function setFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!ALLOWED_EXTS.includes(ext)) {
    alert(`Invalid file type. Allowed: ${ALLOWED_EXTS.join(', ')}`);
    return;
  }

  currentFile = file;

  // Show preview
  dropZoneInner.style.display = 'none';
  filePreview.style.display = 'block';
  fileName.textContent = file.name;
  fileSize.textContent = window.VoiceIQ?.formatFileSize(file.size) || `${file.size} bytes`;

  // Set audio player source
  const url = URL.createObjectURL(file);
  audioPlayer.src = url;

  analyzeBtn.disabled = false;

  // Hide previous results
  resultsSection.style.display = 'none';
}

function clearFile() {
  currentFile = null;
  audioInput.value = '';
  audioPlayer.src = '';
  dropZoneInner.style.display = 'block';
  filePreview.style.display = 'none';
  analyzeBtn.disabled = true;
  resultsSection.style.display = 'none';
}


// ─── ANALYSIS ─────────────────────────────────

analyzeBtn?.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!currentFile) return;

  // Show loading
  loadingOverlay.style.display = 'block';
  resultsSection.style.display = 'none';
  analyzeBtn.disabled = true;

  // Cycle through loading steps
  const loadingStep = document.getElementById('loadingStep');
  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    loadingStep.textContent = LOADING_STEPS[stepIdx % LOADING_STEPS.length];
    stepIdx++;
  }, 900);

  try {
    const formData = new FormData();
    formData.append('audio', currentFile);

    const response = await fetch('/api/predict/upload', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();

    if (!response.ok || result.error) {
      alert(`Error: ${result.error || 'Prediction failed'}`);
      return;
    }

    lastResult = result;
    renderResults(result);

  } catch (err) {
    console.error('Analysis failed:', err);
    alert('Network error — make sure the Flask server is running.');
  } finally {
    clearInterval(stepInterval);
    loadingOverlay.style.display = 'none';
    analyzeBtn.disabled = false;
  }
}


// ─── RENDER RESULTS ───────────────────────────

function renderResults(data) {
  // Primary emotion card
  document.getElementById('resultEmoji').textContent = data.emotion_emoji || '🎭';
  document.getElementById('resultEmotion').textContent = data.emotion || '—';
  document.getElementById('resultConfidence').textContent = data.confidence || '0';
  document.getElementById('resultInferenceTime').textContent =
    data.inference_ms ? `⚡ ${data.inference_ms}ms inference` : '';

  // Color the hero card border by emotion
  const heroCard = document.getElementById('emotionCard');
  heroCard.style.borderColor = data.emotion_color || '';

  // Secondary cards
  document.getElementById('resultGender').textContent = data.gender || '—';
  document.getElementById('resultAge').textContent = data.age_group || '—';

  const attrs = data.speech_attributes || {};
  document.getElementById('resultEnergy').textContent =
    attrs.energy_level ? attrs.energy_level.charAt(0).toUpperCase() + attrs.energy_level.slice(1) : '—';
  document.getElementById('resultDuration').textContent =
    attrs.duration_sec ? `${attrs.duration_sec}s` : '—';

  // Waveform
  renderWaveform(data.waveform || []);

  // Emotion bar chart
  renderEmotionChart(data.emotion_probabilities || []);

  // Speech attributes detail
  renderAttrs(attrs);

  // Show results
  resultsSection.style.display = 'flex';
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderWaveform(points) {
  const ctx = document.getElementById('waveformChart').getContext('2d');

  if (waveformChart) waveformChart.destroy();

  waveformChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: points.map((_, i) => i),
      datasets: [{
        data: points,
        borderColor: '#5b8ef0',
        borderWidth: 1.5,
        fill: true,
        backgroundColor: (context) => {
          const chart = context.chart;
          const { ctx: c, chartArea } = chart;
          if (!chartArea) return 'transparent';
          const gradient = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
          gradient.addColorStop(0, 'rgba(91,142,240,0.3)');
          gradient.addColorStop(1, 'rgba(91,142,240,0)');
          return gradient;
        },
        tension: 0.3,
        pointRadius: 0,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { display: false, min: -1, max: 1 },
      },
      animation: { duration: 600, easing: 'easeInOutCubic' },
    },
  });
}

function renderEmotionChart(emotionData) {
  const ctx = document.getElementById('emotionChart').getContext('2d');

  if (emotionChart) emotionChart.destroy();

  const COLORS = {
    neutral:   '#94a3b8',
    calm:      '#67e8f9',
    happy:     '#fbbf24',
    sad:       '#60a5fa',
    angry:     '#f87171',
    fearful:   '#a78bfa',
    disgust:   '#86efac',
    surprised: '#fb923c',
  };

  const labels = emotionData.map(([label]) => label);
  const values = emotionData.map(([, val]) => val);
  const bgColors = labels.map(l => COLORS[l] || '#5b8ef0');

  emotionChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: bgColors.map(c => c + '90'),
        borderColor: bgColors,
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.raw}%`,
          },
        },
      },
      scales: {
        x: {
          max: 100,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#8b9bbe', font: { family: 'JetBrains Mono', size: 11 }, callback: v => v + '%' },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#f0f4ff', font: { family: 'Syne', size: 12, weight: '600' } },
        },
      },
      animation: { duration: 800, easing: 'easeInOutCubic' },
    },
  });
}

function renderAttrs(attrs) {
  const grid = document.getElementById('attrsGrid');
  const items = [
    { label: 'Pitch (Hz)',      value: attrs.pitch_hz ?? '—' },
    { label: 'Tempo (syl/s)',   value: attrs.speech_tempo ?? '—' },
    { label: 'RMS Energy',      value: attrs.energy_rms ?? '—' },
    { label: 'Confidence',      value: attrs.confidence_score != null ? (attrs.confidence_score * 100).toFixed(0) + '%' : '—' },
    { label: 'Energy Level',    value: attrs.energy_level ?? '—' },
    { label: 'Duration',        value: attrs.duration_sec != null ? attrs.duration_sec + 's' : '—' },
  ];

  grid.innerHTML = items.map(({ label, value }) => `
    <div class="attr-item">
      <div class="attr-label">${label}</div>
      <div class="attr-value">${value}</div>
    </div>
  `).join('');
}


// ─── DOWNLOAD REPORT ─────────────────────────

document.getElementById('downloadReport')?.addEventListener('click', () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `voiceiq-report-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

document.getElementById('analyzeAnother')?.addEventListener('click', () => {
  clearFile();
  resultsSection.style.display = 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
