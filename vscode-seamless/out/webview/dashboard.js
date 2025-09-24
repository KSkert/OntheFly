"use strict";
// This file is bundled by VS Code automatically (no build step needed if kept simple)
/// <reference lib="dom" />
const vscode = acquireVsCodeApi();
const byId = (id) => document.getElementById(id);
const btnChoose = byId('btnChoose');
const btnSetPy = byId('btnSetPy');
const pyPath = byId('pyPath');
const btnStart = byId('btnStart');
const btnStop = byId('btnStop');
const btnPause = byId('btnPause');
const btnResume = byId('btnResume');
const btnRewind = byId('btnRewind');
const rewindSteps = byId('rewindSteps');
const btnSaveCkpt = byId('btnSaveCkpt');
const btnFork = byId('btnFork');
const btnMerge = byId('btnMerge');
const btnAutoSave = byId('btnAutoSave');
const btnLoad = byId('btnLoad');
const modeSel = byId('modeSel');
const runSel = byId('runSel');
const fsSel = byId('fsSel');
const scriptName = byId('scriptName');
const logDiv = byId('log');
let lossChart, gurChart, smoothChart;
function send(command, extra = {}) {
    vscode.postMessage({ command, ...extra });
}
btnChoose.onclick = () => send('chooseScript');
btnSetPy.onclick = () => send('setPython', { path: pyPath.value || 'python' });
btnStart.onclick = () => send('start', { mode: modeSel.value });
btnStop.onclick = () => send('stop');
btnPause.onclick = () => send('pause');
btnResume.onclick = () => send('resume');
btnRewind.onclick = () => send('rewind', { steps: Number(rewindSteps.value || 0) });
btnSaveCkpt.onclick = () => send('saveCkpt');
btnFork.onclick = () => send('fork', { payload: { hparams: { lr_mul: 0.5 } } });
btnMerge.onclick = async () => {
    const paths = prompt('Enter two checkpoint paths separated by comma');
    if (!paths)
        return;
    const list = paths.split(',').map(s => s.trim()).filter(Boolean);
    send('merge', { payload: { paths: list, strategy: 'swa' } });
};
btnAutoSave.onclick = () => send('saveSessionAs');
btnLoad.onclick = () => send('loadSession');
modeSel.onchange = () => send('setMode', { mode: modeSel.value });
byId('btnRefreshRuns').onclick = () => send('requestRuns');
runSel.onchange = () => {
    const runId = runSel.value;
    send('requestFeatureSets', { runId });
};
fsSel.onchange = () => {
    const runId = runSel.value;
    const featureSet = fsSel.value;
    send('requestFeatureSetRows', { runId, featureSet });
};
initCharts();
window.addEventListener('message', (e) => {
    const m = e.data;
    switch (m.type) {
        case 'scriptChosen':
            scriptName.textContent = `Selected: ${m.file}`;
            break;
        case 'newRun':
            send('requestRuns');
            break;
        case 'runs':
            fillRunSel(m.rows);
            break;
        case 'featureSets':
            fillFsSel(m.sets);
            break;
        case 'featureSetRows': {
            const rows = m.rows || [];
            const steps = rows.map((r) => r.step);
            const loss = rows.map((r) => r.loss);
            const gur = rows.map((r) => r.gur);
            const theta = rows.map((r) => r.theta);
            const c = rows.map((r) => r.c);
            drawCharts(steps, loss, gur, theta, c);
            break;
        }
        case 'metric': {
            // live streaming
            appendMetric(m.payload);
            break;
        }
        case 'checkpointSaved':
        case 'paused':
        case 'resumed':
        case 'epoch_end':
        case 'auto_fork_suggested':
        case 'merged':
        case 'forked':
            log(`▶ ${JSON.stringify(m)}`);
            break;
        case 'trainingFinished':
            log('Training finished.');
            break;
        case 'sessionLoaded':
            log('Session loaded. Refreshing runs...');
            send('requestRuns');
            break;
        case 'log':
            log(m.text);
            break;
        case 'error':
            log(`[stderr] ${m.text}`);
            break;
    }
});
function fillRunSel(rows) {
    runSel.innerHTML = '';
    rows.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.run_id;
        opt.textContent = `${r.run_id} (${r.mode}) ${r.name}`;
        runSel.appendChild(opt);
    });
    if (rows.length) {
        runSel.value = rows[0].run_id;
        send('requestFeatureSets', { runId: runSel.value });
    }
}
function fillFsSel(sets) {
    fsSel.innerHTML = '';
    sets.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = s;
        fsSel.appendChild(opt);
    });
    if (sets.length) {
        fsSel.value = sets[0];
        send('requestFeatureSetRows', { runId: runSel.value, featureSet: fsSel.value });
    }
}
/* ---------------- charts ---------------- */
function initCharts() {
    const LC = document.getElementById('lossChart').getContext('2d');
    lossChart = new window.Chart(LC, {
        type: 'line',
        data: { labels: [], datasets: [
                { label: 'Loss', data: [], borderColor: 'blue', fill: false }
            ] },
        options: { animation: false, elements: { point: { radius: 0 } } }
    });
    const GC = document.getElementById('gurChart').getContext('2d');
    gurChart = new window.Chart(GC, {
        type: 'line',
        data: { labels: [], datasets: [
                { label: 'GUR', data: [], borderColor: 'orange', fill: false }
            ] },
        options: { animation: false, elements: { point: { radius: 0 } } }
    });
    const SC = document.getElementById('smoothChart').getContext('2d');
    smoothChart = new window.Chart(SC, {
        type: 'line',
        data: { labels: [], datasets: [
                { label: 'theta', data: [], borderColor: 'red', fill: false },
                { label: 'c', data: [], borderColor: 'purple', fill: false }
            ] },
        options: { animation: false, elements: { point: { radius: 0 } } }
    });
}
function drawCharts(steps, loss, gur, theta, c) {
    lossChart.data.labels = steps;
    lossChart.data.datasets[0].data = loss;
    lossChart.update('none');
    gurChart.data.labels = steps;
    gurChart.data.datasets[0].data = gur;
    gurChart.update('none');
    smoothChart.data.labels = steps;
    smoothChart.data.datasets[0].data = theta;
    smoothChart.data.datasets[1].data = c;
    smoothChart.update('none');
}
function appendMetric(p) {
    const s = p.step;
    lossChart.data.labels.push(s);
    gurChart.data.labels.push(s);
    smoothChart.data.labels.push(s);
    lossChart.data.datasets[0].data.push(p.loss ?? null);
    gurChart.data.datasets[0].data.push(p.gur ?? null);
    smoothChart.data.datasets[0].data.push(p.theta ?? null);
    smoothChart.data.datasets[1].data.push(p.c ?? null);
    lossChart.update('none');
    gurChart.update('none');
    smoothChart.update('none');
}
function log(t) {
    logDiv.textContent += t + '\n';
}
//# sourceMappingURL=dashboard.js.map