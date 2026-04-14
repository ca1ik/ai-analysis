/* ═══════════════════════════════════════════════════════════
   AI Training Command Center — Client App
   Real-time dashboard, charts, chat, training control
   ═══════════════════════════════════════════════════════════ */

const API = '';  // Same origin
let lossChart, gpuChart;
let gpuHistory = { temps: [], powers: [], utils: [], labels: [] };
const GPU_HISTORY_MAX = 60;

// ─── Page Navigation ─────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const page = item.dataset.page;
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        item.classList.add('active');
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        const target = document.getElementById(`page-${page}`);
        if (target) target.classList.add('active');
        if (page === 'logs') refreshLogs();
        if (page === 'info') loadInfrastructure();
    });
});

// ─── Init Charts ─────────────────────────────────────────
function initCharts() {
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
        },
        scales: {
            x: {
                grid: { color: 'rgba(255,255,255,0.04)' },
                ticks: { color: '#555570', font: { size: 10, family: "'JetBrains Mono'" } }
            },
            y: {
                grid: { color: 'rgba(255,255,255,0.04)' },
                ticks: { color: '#555570', font: { size: 10, family: "'JetBrains Mono'" } }
            }
        }
    };

    // Loss Chart
    const lossCtx = document.getElementById('lossChart');
    if (lossCtx) {
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    data: [],
                    borderColor: '#6c5ce7',
                    backgroundColor: 'rgba(108,92,231,0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: '#6c5ce7',
                    tension: 0.3,
                    fill: true,
                }]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: { ...chartDefaults.scales.y, beginAtZero: false }
                }
            }
        });
    }

    // GPU Chart (multi-line)
    const gpuCtx = document.getElementById('gpuChart');
    if (gpuCtx) {
        gpuChart = new Chart(gpuCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Util %',
                        data: [],
                        borderColor: '#00b894',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                    },
                    {
                        label: 'Temp °C',
                        data: [],
                        borderColor: '#feca57',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                    },
                    {
                        label: 'Power %',
                        data: [],
                        borderColor: '#a29bfe',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                    }
                ]
            },
            options: {
                ...chartDefaults,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#8888a8',
                            font: { size: 11, family: "'Inter'" },
                            boxWidth: 12,
                            padding: 16,
                        }
                    }
                },
                scales: {
                    ...chartDefaults.scales,
                    y: {
                        ...chartDefaults.scales.y,
                        min: 0,
                        max: 100,
                    }
                }
            }
        });
    }
}

// ─── Update Functions ────────────────────────────────────
function updateLevel(level) {
    if (!level) return;

    // Hero level
    const heroLevel = document.getElementById('heroLevel');
    const heroTier = document.getElementById('heroTier');
    const heroRing = document.getElementById('heroRing');
    const heroXpFill = document.getElementById('heroXpFill');
    const heroStep = document.getElementById('heroStep');
    const heroProgress = document.getElementById('heroProgress');

    if (heroLevel) heroLevel.textContent = level.level;
    if (heroTier) {
        heroTier.textContent = level.tier;
        heroTier.style.color = level.tier_color;
    }
    if (heroRing) {
        const circumference = 2 * Math.PI * 85;
        const offset = circumference - (level.progress / 100) * circumference;
        heroRing.style.strokeDashoffset = offset;
        heroRing.style.stroke = level.tier_color;
    }
    if (heroXpFill) heroXpFill.style.width = level.xp_progress + '%';
    if (heroStep) heroStep.textContent = level.current_step.toLocaleString();
    if (heroProgress) heroProgress.textContent = level.progress + '%';

    // Sidebar level
    const levelNumber = document.getElementById('levelNumber');
    const levelTier = document.getElementById('levelTier');
    const levelRing = document.getElementById('levelRing');
    const levelProgressText = document.getElementById('levelProgressText');

    if (levelNumber) {
        levelNumber.textContent = level.level;
        levelNumber.style.color = level.tier_color;
    }
    if (levelTier) {
        levelTier.textContent = level.tier;
        levelTier.style.color = level.tier_color;
    }
    if (levelRing) {
        const circumference = 2 * Math.PI * 52;
        const offset = circumference - (level.progress / 100) * circumference;
        levelRing.style.strokeDashoffset = offset;
        levelRing.style.stroke = level.tier_color;
    }
    if (levelProgressText) levelProgressText.textContent = `${level.progress}% Complete`;
}

function updateGPU(gpu) {
    if (!gpu) return;

    const util = gpu.utilization || 0;
    const temp = gpu.temperature || 0;
    const power = gpu.power_draw || 0;
    const powerLimit = gpu.power_limit || 250;
    const memUsed = gpu.memory_used || 0;
    const memTotal = gpu.memory_total || 12227;
    const powerPct = powerLimit > 0 ? (power / powerLimit * 100) : 0;
    const memPct = memTotal > 0 ? (memUsed / memTotal * 100) : 0;

    setText('gpuUtil', util + '%');
    setText('gpuTemp', temp + '°C');
    setText('gpuPower', Math.round(power) + 'W / ' + Math.round(powerLimit) + 'W');
    setText('gpuVram', Math.round(memUsed) + ' / ' + Math.round(memTotal) + ' MiB');

    setWidth('gpuUtilBar', util);
    setWidth('gpuTempBar', temp);  // Assume 90°C max
    setWidth('gpuPowerBar', powerPct);
    setWidth('gpuVramBar', memPct);

    // Mini GPU in sidebar
    const gpuMiniFill = document.getElementById('gpuMiniFill');
    if (gpuMiniFill) gpuMiniFill.style.width = util + '%';
    setText('gpuMiniTemp', temp + '°C');

    // Color temp bar based on temperature
    const tempBar = document.getElementById('gpuTempBar');
    if (tempBar) {
        if (temp > 80) tempBar.style.background = '#ff6b6b';
        else if (temp > 65) tempBar.style.background = '#feca57';
        else tempBar.style.background = '#00b894';
    }

    // GPU History for chart
    const now = new Date();
    const label = now.getHours().toString().padStart(2, '0') + ':' +
                  now.getMinutes().toString().padStart(2, '0') + ':' +
                  now.getSeconds().toString().padStart(2, '0');

    gpuHistory.labels.push(label);
    gpuHistory.utils.push(util);
    gpuHistory.temps.push(temp);
    gpuHistory.powers.push(Math.round(powerPct));

    if (gpuHistory.labels.length > GPU_HISTORY_MAX) {
        gpuHistory.labels.shift();
        gpuHistory.utils.shift();
        gpuHistory.temps.shift();
        gpuHistory.powers.shift();
    }

    if (gpuChart) {
        gpuChart.data.labels = gpuHistory.labels;
        gpuChart.data.datasets[0].data = gpuHistory.utils;
        gpuChart.data.datasets[1].data = gpuHistory.temps;
        gpuChart.data.datasets[2].data = gpuHistory.powers;
        gpuChart.update('none');
    }
}

function updateTrainingStatus(training) {
    if (!training) return;

    const badge = document.getElementById('trainingBadge');
    if (badge) {
        if (training.running) {
            badge.textContent = 'TRAINING';
            badge.className = 'status-badge running';
        } else {
            badge.textContent = 'IDLE';
            badge.className = 'status-badge';
        }
    }

    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');
    if (btnStart) btnStart.disabled = training.running;
    if (btnStop) btnStop.disabled = !training.running;

    const progressDiv = document.getElementById('trainingProgress');
    if (progressDiv) {
        progressDiv.style.display = training.current_step > 0 ? 'block' : 'none';
    }

    const total = training.total_steps || 1701;
    const current = training.current_step || 0;
    const pct = total > 0 ? (current / total * 100) : 0;

    setWidth('progressFill', pct);
    setText('progressText', `${current} / ${total} steps (${pct.toFixed(1)}%)`);
    setText('progStarted', training.started_at || '--');
    setText('progChunks', training.chunks_completed || '0');
    setText('progErrors', training.errors || '0');
}

function updateCheckpoints(checkpoints) {
    const container = document.getElementById('checkpointsList');
    if (!container || !checkpoints) return;

    if (checkpoints.length === 0) {
        container.innerHTML = '<div class="empty-state">No checkpoints yet</div>';
        return;
    }

    container.innerHTML = checkpoints.map(c => {
        const date = new Date(c.created);
        const timeStr = date.toLocaleTimeString('tr-TR');
        const dateStr = date.toLocaleDateString('tr-TR');
        return `
            <div class="checkpoint-item">
                <span class="checkpoint-step">Step ${c.step}</span>
                <div class="checkpoint-meta">
                    <span>${c.size_mb} MB</span>
                    <span>${c.loss !== null ? 'Loss: ' + c.loss.toFixed(4) : ''}</span>
                    <span>${dateStr} ${timeStr}</span>
                </div>
            </div>
        `;
    }).join('');
}

function updateLossChart(metrics) {
    if (!lossChart || !metrics || !metrics.steps || metrics.steps.length === 0) return;

    lossChart.data.labels = metrics.steps.map(s => 'Step ' + s);
    lossChart.data.datasets[0].data = metrics.losses;
    lossChart.update();
}

// ─── Helpers ─────────────────────────────────────────────
function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function setWidth(id, pct) {
    const el = document.getElementById(id);
    if (el) el.style.width = Math.min(100, Math.max(0, pct)) + '%';
}

// ─── API Calls ───────────────────────────────────────────
async function fetchStatus() {
    try {
        const resp = await fetch(API + '/api/status');
        if (!resp.ok) return;
        const data = await resp.json();

        updateLevel(data.level);
        updateGPU(data.gpu);
        updateTrainingStatus(data.training);
        updateCheckpoints(data.checkpoints);

        const heroLoss = document.getElementById('heroLoss');
        if (heroLoss && data.checkpoints && data.checkpoints.length > 0) {
            const last = data.checkpoints[data.checkpoints.length - 1];
            if (last.loss !== null) heroLoss.textContent = last.loss.toFixed(4);
        }

        setText('lastUpdate', new Date().toLocaleTimeString('tr-TR'));
    } catch (e) {
        console.warn('Status fetch failed:', e);
    }
}

async function fetchMetrics() {
    try {
        const resp = await fetch(API + '/api/metrics');
        if (!resp.ok) return;
        const data = await resp.json();
        updateLossChart(data);
    } catch (e) {
        console.warn('Metrics fetch failed:', e);
    }
}

async function refreshLogs() {
    try {
        const resp = await fetch(API + '/api/logs?lines=300');
        if (!resp.ok) return;
        const data = await resp.json();
        const logEl = document.getElementById('logOutput');
        if (logEl) {
            logEl.textContent = data.logs.join('\n');
            logEl.parentElement.scrollTop = logEl.parentElement.scrollHeight;
        }
    } catch (e) {
        console.warn('Logs fetch failed:', e);
    }
}

async function loadInfrastructure() {
    try {
        const resp = await fetch(API + '/api/infrastructure');
        if (!resp.ok) return;
        const data = await resp.json();

        const grid = document.getElementById('infoGrid');
        if (!grid) return;

        grid.innerHTML = '';

        const sections = [
            { icon: '💻', title: 'System', data: data.system },
            { icon: '🖥️', title: 'GPU', data: data.gpu },
            { icon: '🧠', title: 'ML Stack', data: data.ml_stack },
            { icon: '🎯', title: 'Model Configuration', data: data.model },
            { icon: '⚡', title: 'Training Strategy', data: data.strategy },
        ];

        sections.forEach(sec => {
            const card = document.createElement('div');
            card.className = 'info-card';
            let rows = '';
            for (const [k, v] of Object.entries(sec.data)) {
                const key = k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                const val = typeof v === 'boolean' ? (v ? '✅ Yes' : '❌ No') : String(v);
                rows += `<div class="info-row"><span class="info-key">${key}</span><span class="info-value">${val}</span></div>`;
            }
            card.innerHTML = `<h3>${sec.icon} ${sec.title}</h3>${rows}`;
            grid.appendChild(card);
        });
    } catch (e) {
        console.warn('Infrastructure fetch failed:', e);
    }
}

// ─── Training Control ────────────────────────────────────
async function startTraining() {
    const config = {
        steps_per_chunk: parseInt(document.getElementById('cfgStepsPerChunk').value),
        total_steps: parseInt(document.getElementById('cfgTotalSteps').value),
        batch_size: parseInt(document.getElementById('cfgBatchSize').value),
        grad_accum: parseInt(document.getElementById('cfgGradAccum').value),
        learning_rate: parseFloat(document.getElementById('cfgLR').value),
        max_seq_length: parseInt(document.getElementById('cfgMaxSeq').value),
        gpu_power_limit: parseInt(document.getElementById('cfgGpuPower').value),
    };

    try {
        const resp = await fetch(API + '/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await resp.json();
        if (resp.ok) {
            showToast('Training started! PID: ' + data.pid, 'success');
        } else {
            showToast('Error: ' + (data.detail || 'Unknown'), 'error');
        }
    } catch (e) {
        showToast('Connection error', 'error');
    }
}

async function stopTraining() {
    if (!confirm('Stop training? Progress is saved in checkpoints.')) return;
    try {
        const resp = await fetch(API + '/api/training/stop', { method: 'POST' });
        const data = await resp.json();
        showToast('Training ' + data.status, 'success');
    } catch (e) {
        showToast('Error stopping training', 'error');
    }
}

// ─── Chat ────────────────────────────────────────────────
const chatInput = document.getElementById('chatInput');
const chatTemp = document.getElementById('chatTemp');
const chatTempVal = document.getElementById('chatTempVal');

if (chatTemp) {
    chatTemp.addEventListener('input', () => {
        if (chatTempVal) chatTempVal.textContent = chatTemp.value;
    });
}

if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });
}

function sendSuggestion(text) {
    if (chatInput) chatInput.value = text;
    sendMessage();
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const messages = document.getElementById('chatMessages');
    const btnSend = document.getElementById('btnSend');
    if (!input || !messages) return;

    const text = input.value.trim();
    if (!text) return;

    // Clear welcome if present
    const welcome = messages.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    // Add user message
    addChatMessage('user', text);
    input.value = '';
    input.style.height = 'auto';

    // Show loading
    const loadingId = 'loading-' + Date.now();
    const loadingHtml = `
        <div class="chat-msg assistant" id="${loadingId}">
            <div class="chat-msg-avatar">🤖</div>
            <div class="chat-msg-bubble">
                <div class="chat-msg-loading"><span></span><span></span><span></span></div>
            </div>
        </div>
    `;
    messages.insertAdjacentHTML('beforeend', loadingHtml);
    messages.scrollTop = messages.scrollHeight;

    if (btnSend) btnSend.disabled = true;

    try {
        const resp = await fetch(API + '/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                max_tokens: parseInt(document.getElementById('chatMaxTokens')?.value || 512),
                temperature: parseFloat(document.getElementById('chatTemp')?.value || 0.7),
            }),
        });

        const loading = document.getElementById(loadingId);
        if (loading) loading.remove();

        if (resp.ok) {
            const data = await resp.json();
            addChatMessage('assistant', data.response);

            const badge = document.getElementById('chatModelBadge');
            if (badge) badge.textContent = data.model;
        } else {
            const err = await resp.json();
            addChatMessage('assistant', '⚠️ ' + (err.detail || 'Error occurred'));
        }
    } catch (e) {
        const loading = document.getElementById(loadingId);
        if (loading) loading.remove();
        addChatMessage('assistant', '⚠️ Connection error. Is the server running?');
    }

    if (btnSend) btnSend.disabled = false;
}

function addChatMessage(role, text) {
    const messages = document.getElementById('chatMessages');
    if (!messages) return;

    const avatar = role === 'user' ? '👤' : '🤖';
    const html = `
        <div class="chat-msg ${role}">
            <div class="chat-msg-avatar">${avatar}</div>
            <div class="chat-msg-bubble">${escapeHtml(text)}</div>
        </div>
    `;
    messages.insertAdjacentHTML('beforeend', html);
    messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

// ─── Toast Notification ──────────────────────────────────
function showToast(msg, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 24px; right: 24px; z-index: 9999;
        padding: 14px 24px; border-radius: 10px;
        font-size: 14px; font-weight: 500; font-family: var(--font);
        color: #fff; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        animation: fadeIn 0.3s ease;
        background: ${type === 'success' ? '#00b894' : type === 'error' ? '#ff6b6b' : '#6c5ce7'};
    `;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ─── Polling Loop ────────────────────────────────────────
initCharts();
fetchStatus();
fetchMetrics();

// Poll status every 3 seconds
setInterval(fetchStatus, 3000);

// Poll metrics every 15 seconds
setInterval(fetchMetrics, 15000);

// Poll logs if on logs page every 5 seconds
setInterval(() => {
    const logsPage = document.getElementById('page-logs');
    if (logsPage && logsPage.classList.contains('active')) {
        refreshLogs();
    }
}, 5000);
