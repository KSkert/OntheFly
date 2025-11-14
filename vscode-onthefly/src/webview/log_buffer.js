/* log_buffer.js
 * Self-contained log ring buffer that mirrors the dashboard textarea.
 */
(function () {
  const MAX_LOG_LINES = 2000;
  const ring = new Array(MAX_LOG_LINES);

  let start = 0;
  let length = 0;
  let flushTimer = null;
  let cachedTarget = null;

  function getTarget() {
    const root = (typeof document !== 'undefined') ? document.body : null;
    if (!root) return null;
    if (cachedTarget && root.contains(cachedTarget)) return cachedTarget;
    cachedTarget = document.getElementById('log');
    return cachedTarget;
  }

  function resetRing() {
    start = 0;
    length = 0;
    ring.fill(undefined);
  }

  function cancelFlushTimer() {
    if (!flushTimer) return;
    clearTimeout(flushTimer);
    flushTimer = null;
  }

  function clearLogs() {
    resetRing();
    cancelFlushTimer();
    const target = getTarget();
    if (!target) return;
    target.value = '';
    target.scrollTop = 0;
  }

  function push(line) {
    const idx = (start + length) % MAX_LOG_LINES;
    ring[idx] = line;
    if (length < MAX_LOG_LINES) {
      length++;
    } else {
      start = (start + 1) % MAX_LOG_LINES;
    }
  }

  function flush() {
    const target = getTarget();
    if (!target) return;
    const out = new Array(length);
    for (let i = 0; i < length; i++) {
      out[i] = ring[(start + i) % MAX_LOG_LINES];
    }
    const atBottom = (target.scrollTop + target.clientHeight) >= (target.scrollHeight - 4);
    target.value = out.join('\n');
    if (atBottom) target.scrollTop = target.scrollHeight;
  }

  function log(input) {
    if (typeof document === 'undefined') return;
    const target = getTarget();
    if (!target) return;
    push(String(input));
    if (!flushTimer) {
      flushTimer = setTimeout(() => {
        flushTimer = null;
        flush();
      }, 200);
    }
  }

  window.LogBuffer = { log, clearLogs };
  window.log = log;
})();
