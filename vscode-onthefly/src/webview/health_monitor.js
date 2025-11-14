(function (global) {
  'use strict';
  
  // ===============================
  // Health Monitor dock (panels)
  // API: enableHealthMonitor(root?, opts?) -> controller
  // ===============================
  const LABELS = {
    btnActivationsHealth: 'Activation Health',
    btnDeterminismHealth: 'Determinism Health',
    btnNumericsHealth: 'Numerics Health',
    btnDistHealth: 'Distribution Health',
    btnThroughputHealth: 'Throughput Health'
  };

  const MIN_PX = 160;   // minimum panel width when dragging
  const HANDLE_PX = 8;  // splitter thickness

  function qs(sel, from = document) { return from.querySelector(sel); }

  function defaultRoot() {
    const panel = qs('#healthMonitorPanel');
    return (panel && panel.closest('.widget')) || qs('[data-widget-id="w-health"]') || panel || null;
  }

  function ensureDock(root) {
    let dock = root && root.querySelector('.hmDock');
    if (!dock && root) {
      dock = document.createElement('div');
      dock.className = 'hmDock';
      dock.setAttribute('role', 'region');
      dock.setAttribute('aria-label', 'Health panels dock');
      (qs('#healthMonitorPanel', root) || root).appendChild(dock);
    }
    return dock;
  }

  function injectStylesOnce() {
    if (qs('#hm-split-css')) return;
    const css = `
      /* splitter layout helper */
      .hmDock.hmr { display:flex; width:100%; height:100%; gap:10px; }
      .hmDock.hmr .hmCard { flex:1 1 0%; display:flex; flex-direction:column; min-width:${MIN_PX}px; }
      .hmDock.hmr .hmHandle { flex:0 0 ${HANDLE_PX}px; cursor:col-resize; user-select:none; position:relative; }
      .hmDock.hmr .hmHandle::after { content:""; position:absolute; top:20%; bottom:20%; left:calc(50% - 1px); width:2px; background:var(--border); border-radius:1px; opacity:.9; }
      @media (pointer:coarse){ .hmDock.hmr .hmHandle{ flex-basis:12px; } }
      `;
    const tag = document.createElement('style');
    tag.id = 'hm-split-css';
    tag.textContent = css;
    try {
      const cur = document.currentScript;
      if (cur && cur.nonce) tag.nonce = cur.nonce;
    } catch {}
    document.head.appendChild(tag);
  }

  function createCard(key, title) {
    const card = document.createElement('article');
    card.className = 'hmCard';
    card.setAttribute('data-key', key);

    const header = document.createElement('div');
    header.className = 'hmCardHeader';

    const ttl = document.createElement('div');
    ttl.className = 'hmTitle';
    ttl.textContent = title;

    const close = document.createElement('button');
    close.className = 'icon-btn ghostBtn hmClose';
    close.setAttribute('aria-label', 'Close ' + title);
    close.textContent = '×';

    const body = document.createElement('div');
    body.className = 'hmBody';
    body.textContent = 'Loading ...';

    header.appendChild(ttl);
    header.appendChild(close);
    card.appendChild(header);
    card.appendChild(body);
    return card;
  }

  function refreshHandles(dock) {
    // remove old
    dock.querySelectorAll('.hmHandle').forEach(h => h.remove());
    const cards = Array.from(dock.querySelectorAll('.hmCard'));
    for (let i = cards.length - 1; i > 0; i--) {
      const handle = document.createElement('div');
      handle.className = 'hmHandle';
      handle.setAttribute('role', 'separator');
      handle.setAttribute('aria-orientation', 'vertical');
      cards[i].before(handle);
    }
  }

  function equalize(dock) {
    const cards = Array.from(dock.querySelectorAll('.hmCard'));
    cards.forEach(c => { c.style.flex = '1 1 0%'; c.removeAttribute('data-px'); });
    refreshHandles(dock);
  }

  function openPanel(dock, key) {
    injectStylesOnce();
    dock.classList.add('hmr');

    const title = LABELS[key] || key;
    let card = dock.querySelector('.hmCard[data-key="'+key+'"]');
    if (!card) {
      card = createCard(key, title);
      dock.appendChild(card);
      equalize(dock);
    }

    requestAnimationFrame(() => {
      card.scrollIntoView({ behavior: 'smooth', inline: 'nearest', block: 'nearest' });
      card.classList.add('is-peek');
      setTimeout(() => card.classList.remove('is-peek'), 420);
    });
  }

  function closePanel(dock, keyOrEl) {
    const card = typeof keyOrEl === 'string'
      ? dock.querySelector('.hmCard[data-key="'+keyOrEl+'"]')
      : (keyOrEl && keyOrEl.closest && keyOrEl.closest('.hmCard'));
    if (!card) return;
    card.remove();
    if (dock.querySelector('.hmCard')) equalize(dock);
    else refreshHandles(dock);
  }

  // ----- Resizing (drag handles) -----
  let drag = null; // { dock,left,right,leftStart,rightStart,totalStart,startX }

  function onHandleDown(e) {
    const handle = e.target.closest('.hmHandle');
    if (!handle) return;

    const d = handle.parentElement;
    const left = handle.previousElementSibling;
    const right = handle.nextElementSibling;
    if (!(left && right) || !left.classList.contains('hmCard') || !right.classList.contains('hmCard')) return;

    const leftW = left.getBoundingClientRect().width;
    const rightW = right.getBoundingClientRect().width;

    drag = {
      dock: d, left, right,
      leftStart: leftW, rightStart: rightW,
      totalStart: leftW + rightW,
      startX: e.clientX
    };

    document.documentElement.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  }

  function onHandleMove(e) {
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    let newLeft = drag.leftStart + dx;
    newLeft = Math.max(MIN_PX, Math.min(newLeft, drag.totalStart - MIN_PX));
    const newRight = drag.totalStart - newLeft;

    const dockW = drag.dock.getBoundingClientRect().width;

    // Left/right fixed px, others flex to fill remaining
    drag.left.style.flex = '0 0 ' + newLeft + 'px';
    drag.right.style.flex = '0 0 ' + newRight + 'px';
    drag.left.dataset.px = String(Math.round(newLeft));
    drag.right.dataset.px = String(Math.round(newRight));
  }

  function onHandleUp() {
    if (!drag) return;
    drag = null;
    document.documentElement.style.cursor = '';
    document.body.style.userSelect = '';
  }

  function onHandleDblClick(e) {
    const h = e.target.closest('.hmHandle');
    if (!h) return;
    const d = h.parentElement;
    equalize(d);
  }

  // ----- Main enabler -----
  function enableHealthMonitor(rootEl, opts = {}) {
    
    const root = rootEl || defaultRoot();
    if (!root) throw new Error('enableHealthMonitor: Health widget/panel not found');
    const d = ensureDock(root);
    if (!d) throw new Error('enableHealthMonitor: .hmDock not found/created');

    injectStylesOnce();
    d.classList.add('hmr');

    // Delegated clicks inside widget
    const onClick = (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;

      // Openers (the five health buttons by ID)
      if (LABELS[btn.id]) {
        openPanel(d, btn.id);

        // force the body to "Loading..." while the extension computes
        const card = d.querySelector(`.hmCard[data-key="${btn.id}"]`);
        if (card) {
          const body = card.querySelector('.hmBody');
          body.textContent = 'Loading...'; // replaces old model's text immediately
          const ttl = card.querySelector('.hmTitle');
          if (ttl) ttl.textContent = (ttl.textContent || '').split(' · ')[0]; // clear any " · OK/Issues"
        }

        return;
      }

      // Closers inside cards
      if (btn.classList.contains('hmClose')) {
        closePanel(d, btn);
      }
    };


    // Attach listeners on the dock and document for handles
    document.addEventListener('click', onClick, true);

    // Object.keys(LABELS).forEach((id) => {
    //   const b = document.getElementById(id);
    //   if (b) b.addEventListener('click', (ev) => { ev.preventDefault(); openPanel(d, id); }, {capture: true});
    // });
    document.addEventListener('mousedown', onHandleDown, true);
    document.addEventListener('mousemove', onHandleMove, true);
    document.addEventListener('mouseup', onHandleUp, true);
    document.addEventListener('dblclick', onHandleDblClick, true);

    // Public controller
    const api = {
      open: (key) => openPanel(d, key),
      close: (keyOrEl) => closePanel(d, keyOrEl),
      equalize: () => equalize(d),
      count: () => d.querySelectorAll('.hmCard').length,
      destroy: () => {
        document.removeEventListener('click', onClick, true);
        document.removeEventListener('mousedown', onHandleDown, true);
        document.removeEventListener('mousemove', onHandleMove, true);
        document.removeEventListener('mouseup', onHandleUp, true);
        document.removeEventListener('dblclick', onHandleDblClick, true);
      }
    };

    // Ensure handles match current DOM
    refreshHandles(d);

    return api;
  }

  // Export in the same style as dag_layout.js
  global.enableHealthMonitor = enableHealthMonitor;
  // Optional namespaced export
  global.OnTheFlyHealthMonitor = { enable: enableHealthMonitor };

  // Convenience auto-enable if panel already in DOM (safe no-op if not found)
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      try { enableHealthMonitor(); } catch {}
    });
  } else {
    try { enableHealthMonitor(); } catch {}
  }

})(window || globalThis);


// ---- Bridge: feed results into the Health cards ----
(function bridgeHealthMessages(global){
  const EVT_TO_BTN = {
    activationsHealth: 'btnActivationsHealth',
    determinismHealth: 'btnDeterminismHealth',
    numericsHealth:    'btnNumericsHealth',
    distHealth:        'btnDistHealth',
    throughputHealth:  'btnThroughputHealth',
  };

  // monospace, wrap newlines nicely
  (function addMonoCss(){
    if (document.getElementById('hm-mono-css')) return;
    const s = document.createElement('style');
    s.id = 'hm-mono-css';
    s.textContent = `.hmMono{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.35;color:var(--vscode-editor-foreground,var(--fg));}`;
    try { const cur=document.currentScript; if (cur && cur.nonce) s.nonce = cur.nonce; } catch{}
    document.head.appendChild(s);
  })();

  function ensureHM(){ return (global.__hm ||= (global.enableHealthMonitor && global.enableHealthMonitor())); }

  // Track which model is currently selected in Model Nav
  let selectedRunId = null;

  function setCardBody(card, text) {
    const body = card.querySelector('.hmBody');
    body.innerHTML = '';
    const pre = document.createElement('pre');
    pre.className = 'hmMono';
    pre.textContent = text;
    body.appendChild(pre);
  }

  function baseTitle(card, key){
    const ttl = card.querySelector('.hmTitle');
    const left = (ttl && ttl.textContent && ttl.textContent.split(' · ')[0]) || (LABELS[key] || key);
    return { ttl, left };
  }

  function openCardFor(eventType){
    const key = EVT_TO_BTN[eventType];
    if (!key) return { key:null, card:null };
    const api = ensureHM(); api && api.open && api.open(key);
    const dock = document.querySelector('.hmDock');
    const card = dock && dock.querySelector(`.hmCard[data-key="${key}"]`);
    return { key, card };
  }

  function handleHealthMessage(m){
    // Ignore results for a different run (prevents old run’s text appearing)
    if (selectedRunId && m.run_id && m.run_id !== selectedRunId) return;

    const { key, card } = openCardFor(m.type);
    if (!card) return;

    const { ttl, left } = baseTitle(card, key);

    if (m.pending) {
      // show during the delay
      setCardBody(card, 'Loading...');
      if (ttl) ttl.textContent = left;
      return;
    }

    if (m.error) {
      setCardBody(card, `Error: ${m.error}`);
      if (ttl) ttl.textContent = `${left} · Error`;
      return;
    }

    // Normal success path
    const data = m.data || {};
    const text = (data.text || data.message) ?? JSON.stringify(data, null, 2);
    setCardBody(card, text);

    if (typeof data.ok === 'boolean') {
      if (ttl) ttl.textContent = `${left} · ${data.ok ? 'OK' : 'Issues'}`;
    } else if (ttl) {
      ttl.textContent = left;
    }
  }

  // Listen for host messages
  window.addEventListener('message', (ev) => {
    const m = ev.data || {};
    if (!m || !m.type) return;

    // Update the currently selected model
    if (m.type === 'modelNav.select') {
      selectedRunId = String(m.runId || '') || null;
      return;
    }

    // Health events (now handle pending/error/data)
    if (EVT_TO_BTN[m.type]) {
      handleHealthMessage(m);
    }
  });

  // also expose for manual calls if you like
  global.OnTheFlyHealthMonitor = Object.assign(
    (global.OnTheFlyHealthMonitor || {}),
    { renderMessage: handleHealthMessage }
  );

})(window || globalThis);
