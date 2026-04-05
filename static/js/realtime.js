/**
 * VoiceIQ — Real-Time Detection Script
 * Handles: MediaRecorder API, live waveform, chunked prediction
 */

const micBtn      = document.getElementById('micBtn');
const micRing     = document.getElementById('micRing');
const micStatus   = document.getElementById('micStatus');
const micTimer    = document.getElementById('micTimer');
const liveResults = document.getElementById('liveResults');
const liveBadge   = document.getElementById('liveBadge');
const permissionError = document.getElementById('permissionError');
const liveWaveCanvas  = document.getElementById('liveWaveCanvas');
const liveHistory     = document.getElementById('liveHistory');
const liveHistoryCount = document.getElementById('liveHistoryCount');

let mediaRecorder = null;
let audioStream   = null;
let audioContext  = null;
let analyserNode  = null;
let animFrameId   = null;
let timerInterval = null;
let recording     = false;
let recordingStart = 0;
let predictionCount = 0;
let liveEmotionChart = null;

const RECORD_DURATION_MS = 3500; // auto-send every 3.5s

const COLORS = {
  neutral:   '#94a3b8', calm: '#67e8f9', happy: '#fbbf24',
  sad:       '#60a5fa', angry: '#f87171', fearful: '#a78bfa',
  disgust:   '#86efac', surprised: '#fb923c',
};


// ─── MIC BUTTON ──────────────────────────────

micBtn.addEventListener('click', async () => {
  if (recording) {
    stopRecording();
  } else {
    await startRecording();
  }
});


// ─── START RECORDING ─────────────────────────

async function startRecording() {
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    console.error('Microphone access denied:', err);
    permissionError.style.display = 'block';
    return;
  }

  recording = true;
  micBtn.classList.add('recording');
  micRing.classList.add('recording');
  micStatus.textContent = 'Recording... speak now!';
  micTimer.style.display = 'block';
  liveBadge.style.display = 'flex';

  document.querySelector('.mic-icon').style.display = 'none';
  document.querySelector('.stop-icon').style.display = 'block';

  recordingStart = Date.now();
  startTimer();
  startWaveformVisualizer(audioStream);
  scheduleRecording(audioStream);
}


// ─── STOP RECORDING ──────────────────────────

function stopRecording() {
  recording = false;
  micBtn.classList.remove('recording');
  micRing.classList.remove('recording');
  micStatus.textContent = 'Click to start recording';
  micTimer.style.display = 'none';
  liveBadge.style.display = 'none';

  document.querySelector('.mic-icon').style.display = 'block';
  document.querySelector('.stop-icon').style.display = 'none';

  clearInterval(timerInterval);
  cancelAnimationFrame(animFrameId);

  if (audioStream) {
    audioStream.getTracks().forEach(t => t.stop());
    audioStream = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  // Clear waveform
  const ctx = liveWaveCanvas.getContext('2d');
  ctx.clearRect(0, 0, liveWaveCanvas.width, liveWaveCanvas.height);

  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
}


// ─── SCHEDULED RECORDING CHUNKS ──────────────

function scheduleRecording(stream) {
  if (!recording) return;

  const chunks = [];

  mediaRecorder = new MediaRecorder(stream, {
    mimeType: getSupportedMimeType(),
  });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    if (chunks.length > 0 && recording) {
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
      await sendAudioBlob(blob, mediaRecorder.mimeType);
    }
    // Schedule next chunk
    if (recording) {
      setTimeout(() => scheduleRecording(stream), 100);
    }
  };

  mediaRecorder.start();

  // Stop current chunk after interval
  setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
  }, RECORD_DURATION_MS);
}


// ─── SEND AUDIO BLOB ─────────────────────────

async function sendAudioBlob(blob, mimeType) {
  try {
    const base64 = await blobToBase64(blob);

    const res = await fetch('/api/predict/realtime', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_blob: base64, mime_type: mimeType }),
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      console.warn('Prediction error:', data.error);
      return;
    }

    updateLiveDisplay(data);
    addToLiveHistory(data);

  } catch (err) {
    console.error('Send audio failed:', err);
  }
}


// ─── UPDATE LIVE DISPLAY ─────────────────────

function updateLiveDisplay(data) {
  liveResults.style.display = 'block';

  // Primary
  document.getElementById('liveEmoji').textContent     = data.emotion_emoji || '🎭';
  document.getElementById('liveEmotion').textContent   = data.emotion || '—';
  document.getElementById('liveConfidence').textContent =
    data.confidence ? `${data.confidence}% confidence` : '';

  // Cards
  document.getElementById('liveGender').textContent  = data.gender || '—';
  document.getElementById('liveAge').textContent     = data.age_group || '—';

  const attrs = data.speech_attributes || {};
  document.getElementById('liveEnergy').textContent  = capitalize(attrs.energy_level || '—');
  document.getElementById('livePitch').textContent   = attrs.pitch_hz ? `${attrs.pitch_hz} Hz` : '—';

  // Emotion probability chart
  renderLiveEmotionChart(data.emotion_probabilities || []);
}


// ─── LIVE EMOTION CHART ───────────────────────

function renderLiveEmotionChart(emotionData) {
  const ctx = document.getElementById('liveEmotionChart').getContext('2d');

  const labels = emotionData.map(([label]) => label);
  const values = emotionData.map(([, val]) => val);
  const bgColors = labels.map(l => COLORS[l] || '#5b8ef0');

  if (liveEmotionChart) {
    liveEmotionChart.data.labels = labels;
    liveEmotionChart.data.datasets[0].data = values;
    liveEmotionChart.data.datasets[0].backgroundColor = bgColors.map(c => c + '90');
    liveEmotionChart.data.datasets[0].borderColor = bgColors;
    liveEmotionChart.update('active');
    return;
  }

  liveEmotionChart = new Chart(ctx, {
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
      animation: { duration: 300 },
      plugins: { legend: { display: false } },
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
    },
  });
}


// ─── LIVE HISTORY ─────────────────────────────

function addToLiveHistory(data) {
  predictionCount++;
  liveHistoryCount.textContent = `${predictionCount} prediction${predictionCount !== 1 ? 's' : ''}`;

  const item = document.createElement('div');
  item.className = 'live-history-item';
  item.innerHTML = `
    <span>${data.emotion_emoji || '🎭'}</span>
    <span style="font-weight:700;text-transform:capitalize;flex:1">${data.emotion || '—'}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--accent)">${data.confidence}%</span>
    <span style="font-size:11px;color:var(--text-muted)">${new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}</span>
  `;

  liveHistory.insertBefore(item, liveHistory.firstChild);

  // Keep only last 10 in view
  while (liveHistory.children.length > 10) {
    liveHistory.removeChild(liveHistory.lastChild);
  }
}


// ─── WAVEFORM VISUALIZER ─────────────────────

function startWaveformVisualizer(stream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 256;

  const source = audioContext.createMediaStreamSource(stream);
  source.connect(analyserNode);

  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  const ctx = liveWaveCanvas.getContext('2d');

  function draw() {
    animFrameId = requestAnimationFrame(draw);
    analyserNode.getByteTimeDomainData(dataArray);

    const w = liveWaveCanvas.width = liveWaveCanvas.offsetWidth;
    const h = liveWaveCanvas.height;

    ctx.clearRect(0, 0, w, h);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#5b8ef0';
    ctx.shadowColor = '#5b8ef0';
    ctx.shadowBlur = 8;
    ctx.beginPath();

    const sliceWidth = w / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * h) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }

    ctx.lineTo(w, h / 2);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  draw();
}


// ─── TIMER ───────────────────────────────────

function startTimer() {
  micTimer.textContent = '00:00';
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - recordingStart) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    micTimer.textContent = `${m}:${s}`;
  }, 1000);
}


// ─── UTILS ───────────────────────────────────

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function getSupportedMimeType() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4'];
  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return '';
}

function capitalize(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : str;
}
