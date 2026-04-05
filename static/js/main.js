/**
 * VoiceIQ — Shared JavaScript Utilities
 * Handles: theme toggle, history loading, global UI helpers
 */

// ─── THEME TOGGLE ────────────────────────────
const themeToggle = document.getElementById('themeToggle');
const html = document.documentElement;

function initTheme() {
  const saved = localStorage.getItem('voiceiq-theme') || 'dark';
  html.setAttribute('data-theme', saved);
}

themeToggle?.addEventListener('click', () => {
  const current = html.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('voiceiq-theme', next);
});

initTheme();


// ─── HISTORY LOADER (index page) ─────────────
async function loadHistory() {
  const historyList = document.getElementById('historyList');
  if (!historyList) return;

  try {
    const res = await fetch('/api/history');
    const data = await res.json();

    if (!data.history || data.history.length === 0) {
      historyList.innerHTML = `
        <div class="history-empty">
          <span>🎙️</span>
          <p>No predictions yet — try uploading an audio file!</p>
        </div>`;
      return;
    }

    const reversed = [...data.history].reverse();
    historyList.innerHTML = reversed.map(item => `
      <div class="history-item">
        <span class="history-emoji">${item.emoji || '🎭'}</span>
        <div class="history-info">
          <div class="history-emotion">${item.emotion || '—'}</div>
          <div class="history-meta">
            ${item.gender || '—'} · ${item.age_group || '—'} · 
            ${item.filename || 'microphone'} · 
            ${formatTime(item.timestamp)}
          </div>
        </div>
        <div class="history-confidence">${item.confidence || 0}%</div>
      </div>
    `).join('');

  } catch (e) {
    console.error('Failed to load history:', e);
  }
}

// ─── CLEAR HISTORY ────────────────────────────
document.getElementById('clearHistory')?.addEventListener('click', async () => {
  if (!confirm('Clear all prediction history?')) return;
  try {
    await fetch('/api/history/clear', { method: 'DELETE' });
    loadHistory();
  } catch (e) {
    console.error('Clear failed:', e);
  }
});


// ─── UTILS ────────────────────────────────────
function formatTime(isoStr) {
  if (!isoStr) return '';
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function capitalize(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : '';
}

// ─── INIT ─────────────────────────────────────
if (document.getElementById('historyList')) {
  loadHistory();
  // Refresh every 10 seconds
  setInterval(loadHistory, 10000);
}

// Export for use in other scripts
window.VoiceIQ = {
  formatTime,
  formatFileSize,
  capitalize,
};
