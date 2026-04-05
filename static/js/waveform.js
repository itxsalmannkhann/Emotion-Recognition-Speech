/**
 * VoiceIQ — Animated Hero Waveform
 * Draws a beautiful animated sine wave on the landing page canvas.
 */

(function () {
  const canvas = document.getElementById('waveCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let frame = 0;

  const WAVES = [
    { freq: 0.025, amp: 18, speed: 0.018, phase: 0,     color: 'rgba(91,142,240,0.8)',  width: 2.5 },
    { freq: 0.040, amp: 12, speed: 0.024, phase: 1.5,   color: 'rgba(162,89,247,0.5)',  width: 1.5 },
    { freq: 0.018, amp: 25, speed: 0.012, phase: 3.0,   color: 'rgba(14,245,200,0.3)',  width: 1.0 },
  ];

  function draw() {
    canvas.width = canvas.offsetWidth || 600;
    const W = canvas.width;
    const H = canvas.height;
    const mid = H / 2;

    ctx.clearRect(0, 0, W, H);

    for (const wave of WAVES) {
      ctx.beginPath();
      ctx.lineWidth = wave.width;
      ctx.strokeStyle = wave.color;
      ctx.shadowColor = wave.color;
      ctx.shadowBlur = 6;

      for (let x = 0; x <= W; x++) {
        const y = mid + Math.sin(x * wave.freq + frame * wave.speed + wave.phase) * wave.amp;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }

      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    frame++;
    requestAnimationFrame(draw);
  }

  draw();
})();
