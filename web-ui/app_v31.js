/**
 * app.js — 약사 AI 챗봇 프론트엔드
 * SSE 스트리밍 · Markdown 렌더링 · 실시간 GPU 모니터링
 */
console.log("%c💊 Pharmacist RAG v2.8 Loaded", "color: #3b82f6; font-weight: bold; font-size: 1.2em;");

/* ══════════════════════════════════════════════════════════
   Configuration
══════════════════════════════════════════════════════════ */
const API_BASE = 'http://localhost:8000';

const EXAMPLE_QUESTIONS = [
    '눈이 침침한데 뭐 먹으면 될까요?',
    '타이레놀과 이부프로펜 같이 먹어도 되나요?',
    '루테인 하루 복용량이 어떻게 되나요?',
    '임산부가 먹어도 되는 영양제가 있나요?',
    '간에 좋은 약은 어떤 게 있나요?',
];

/* ══════════════════════════════════════════════════════════
   State
══════════════════════════════════════════════════════════ */
let isStreaming = false;
let selfCorrEnabled = true;
let currentController = null;
let placeholderTimer = null;

const PLACEHOLDERS = [
    '복용 중인 약 조합이 괜찮은지 물어보세요…',
    '증상별 일반의약품 선택 기준을 질문해보세요…',
    '영양제와 약물 상호작용을 확인해보세요…',
    '부작용 발생 시 대처 방법을 물어보세요…',
];

/* ══════════════════════════════════════════════════════════
   DOM refs
══════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const chatArea = $('chatArea');
const messages = $('messages');
const emptyState = $('emptyState');
const queryInput = $('queryInput');
const sendBtn = $('sendBtn');
const sendIcon = $('sendIcon');
const stopIcon = $('stopIcon');
const statusText = $('statusText');
const metricsOverlay = $('metricsOverlay');
const metricsOverlayText = $('metricsOverlayText');
const bm25Slider = $('bm25Slider');
const bm25Val = $('bm25Val');
const vecVal = $('vecVal');
const vecFill = $('vecFill');
const topkSlider = $('topkSlider');
const topkVal = $('topkVal');
const modelSelect = $('modelSelect');

function updateSendButtonState() {
    if (!sendBtn) return;
    const hasText = queryInput.value.trim().length > 0;
    sendBtn.disabled = !isStreaming && !hasText;
    sendBtn.classList.toggle('ready', hasText && !isStreaming);
}

function startPlaceholderRotation() {
    if (!queryInput) return;
    let idx = 0;
    placeholderTimer = setInterval(() => {
        if (document.activeElement === queryInput) return;
        if (queryInput.value.trim()) return;
        idx = (idx + 1) % PLACEHOLDERS.length;
        queryInput.setAttribute('placeholder', PLACEHOLDERS[idx]);
    }, 4500);
}

/* ══════════════════════════════════════════════════════════
   Markdown → HTML  (lightweight)
══════════════════════════════════════════════════════════ */
function renderMarkdown(text) {
    let html = text
        // escape
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        // bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // italic
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // h3
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        // h2
        .replace(/^## (.+)$/gm, '<h2 style="font-size:1em;margin:.5em 0 .2em;color:var(--text-primary)">$1</h2>')
        // unordered list
        .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
        // ordered list
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        // hr
        .replace(/^---$/gm, '<hr style="border-color:var(--border);margin:.6em 0">')
        // newlines → <p>
        .split(/\n\n+/).map(block => {
            block = block.trim();
            if (!block) return '';
            if (block.startsWith('<h') || block.startsWith('<hr')) return block;
            if (block.startsWith('<li>')) return `<ul>${block}</ul>`;
            return `<p>${block.replace(/\n/g, '<br>')}</p>`;
        }).join('');
    return html;
}

function getExternalApiStatusView(status) {
    const labelMap = {
        openfda: 'OpenFDA',
        mfds_ezdrug: 'MFDS e약은요',
        korea_hybrid: 'Korea Hybrid',
    };

    const shortState = (s) => {
        if (s?.connected === true) return '연결됨';
        if (s?.connected === false) return `오류(${s?.message || 'request_failed'})`;
        return '확인중';
    };

    if (!status) {
        return { text: '🌐 외부 API 상태: 정보 없음', tone: 'unknown' };
    }

    if (status.provider === 'korea_hybrid' && status.providers) {
        const mf = status.providers.mfds_ezdrug || {};
        const of = status.providers.openfda || {};
        const tone = status.connected === true ? 'ok' : status.connected === false ? 'bad' : 'unknown';
        return {
            text: `🌐 외부 API 상태: MFDS ${shortState(mf)} · OpenFDA ${shortState(of)}`,
            tone,
        };
    }

    const label = labelMap[status.provider] || status.provider || 'External API';
    if (status.connected === true) {
        const suffix = status.message === 'connected_no_results' ? '연결됨(결과 없음)' : '연결됨';
        return { text: `🌐 외부 API 상태: ${label} ${suffix}`, tone: 'ok' };
    }
    if (status.connected === false) {
        const reason = status.http_status
            ? `${status.message || 'request_failed'} / HTTP ${status.http_status}`
            : (status.message || 'request_failed');
        return { text: `🌐 외부 API 상태: ${label} 연결 실패 (${reason})`, tone: 'bad' };
    }
    return { text: `🌐 외부 API 상태: ${label} 확인 중...`, tone: 'unknown' };
}

function updateExternalApiStatus(status) {
    const el = $('externalApiStatus');
    if (!el) return;
    const view = getExternalApiStatusView(status);
    el.textContent = view.text;
    el.classList.remove('ok', 'warn', 'bad', 'unknown');
    el.classList.add(view.tone || 'unknown');
}

/* ══════════════════════════════════════════════════════════
   VRAM Polling
══════════════════════════════════════════════════════════ */
async function pollVRAM() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error('Server down');
        const data = await res.json();

        updateVramUI(data.gpu);
        updateExternalApiStatus(data.external_api);

        // Initialization handling
        const overlay = $('initOverlay');
        if (data.status === 'ready') {
            if (!overlay.classList.contains('hidden')) {
                overlay.classList.add('hidden');
                console.log("System Ready - Initialization Overlay Hidden");
            }
        } else {
            overlay.classList.remove('hidden');
            $('initStatusText').textContent = '의약품 지식 베이스 구축 중...';
            if (data.init_logs) {
                const logContainer = $('initLogs');
                logContainer.innerHTML = data.init_logs.map(log => `<div>${log}</div>`).join('');
                logContainer.scrollTop = logContainer.scrollHeight;
            }
        }
    } catch (err) {
        $('vramLabel').textContent = '서버 연결 대기 중...';
        updateExternalApiStatus({ provider: 'external', connected: false, message: 'server_unreachable' });
        console.warn("Polling failed:", err);
    }
}

function updateVramUI(gpu) {
    const dot = $('vramDot');
    const label = $('vramLabel');
    const fill = $('vramFill');
    const stats = $('vramStats');

    if (!gpu.available) {
        dot.className = 'vram-dot no-gpu';
        label.textContent = '⚠️ GPU 미인식 – CPU 모드';
        fill.style.width = '0%';
        fill.style.background = 'var(--text-muted)';
        stats.textContent = '';
    } else {
        const pct = gpu.pct;
        dot.className = 'vram-dot ' + (pct < 60 ? 'gpu-ok' : pct < 85 ? 'gpu-warn' : 'gpu-hot');
        fill.style.width = `${pct}%`;
        fill.style.background = pct < 60
            ? 'var(--green)' : pct < 85 ? 'var(--yellow)' : 'var(--red)';

        // VRAM 텍스트를 더 명확하게 표시 (전용 GPU 메모리 기준)
        label.textContent = `${gpu.name} (${pct.toFixed(1)}%)`;
        stats.textContent = `${gpu.used_gb.toFixed(2)} GB / ${gpu.total_gb.toFixed(2)} GB (전체 점유: ${gpu.reserved_gb.toFixed(2)} GB)`;
    }
}

async function clearMemory() {
    try {
        const btn = $('clearVramBtn');
        const fill = $('vramFill');
        btn.disabled = true;
        btn.style.opacity = '0.5';

        const res = await fetch(`${API_BASE}/clear-memory`, { method: 'POST' });
        const data = await res.json();
        updateVramUI(data.gpu);

        // 시각적 피드백: 바를 잠깐 깜빡임
        if (fill) {
            fill.style.filter = 'brightness(2)';
            setTimeout(() => {
                fill.style.filter = 'none';
            }, 300);
        }

        btn.disabled = false;
        btn.style.opacity = '1';
    } catch (err) {
        console.error('Clear memory failed:', err);
    }
}

pollVRAM();
setInterval(pollVRAM, 3000); // 4초 -> 3초로 단축

/* ══════════════════════════════════════════════════════════
   Sidebar Controls
══════════════════════════════════════════════════════════ */
bm25Slider.addEventListener('input', () => {
    const v = parseInt(bm25Slider.value) / 100;
    const vec = Math.round((1 - v) * 100) / 100;
    bm25Val.textContent = v.toFixed(2);
    vecVal.textContent = vec.toFixed(2);
    vecFill.style.width = `${(1 - v) * 100}%`;
});

topkSlider.addEventListener('input', () => {
    topkVal.textContent = topkSlider.value;
});

function toggleSelfCorr() {
    selfCorrEnabled = !selfCorrEnabled;
    const sw = $('selfCorrSwitch');
    sw.classList.toggle('active', selfCorrEnabled);
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (window.innerWidth <= 640) {
        sidebar.classList.toggle('mobile-open');
    } else {
        sidebar.classList.toggle('collapsed');
    }
}

/* ══════════════════════════════════════════════════════════
   Example Questions
══════════════════════════════════════════════════════════ */
function buildExamplePills() {
    const container = $('examplePills');
    EXAMPLE_QUESTIONS.forEach(q => {
        const btn = document.createElement('button');
        btn.className = 'example-pill';
        btn.innerHTML = `
      <span>${q}</span>
      <svg class="pill-arrow" width="12" height="12" viewBox="0 0 24 24"
           fill="none" stroke="currentColor" stroke-width="2">
        <path d="m9 18 6-6-6-6"/>
      </svg>`;
        btn.onclick = () => {
            queryInput.value = q;
            autoResize();
            sendMessage();
        };
        container.appendChild(btn);
    });
}
buildExamplePills();

/* ══════════════════════════════════════════════════════════
   Input Auto-resize
══════════════════════════════════════════════════════════ */
function autoResize() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 180) + 'px';
    updateSendButtonState();
}
queryInput.addEventListener('input', autoResize);
queryInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

/* ══════════════════════════════════════════════════════════
   Chat utilities
══════════════════════════════════════════════════════════ */
function showEmpty(show) {
    emptyState.classList.toggle('hidden', !show);
    messages.classList.toggle('hidden', show);
}

function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
}

function setStatus(icon, text) {
    statusText.innerHTML = text
        ? `<div class="status-spinner"></div><span>${icon} ${text}</span>`
        : '';
}

function clearStatus() {
    statusText.innerHTML = '';
}

function showMetricsOverlay(message = '작업을 준비하고 있어요…') {
    if (!metricsOverlay) return;
    if (metricsOverlayText) metricsOverlayText.textContent = message;
    metricsOverlay.classList.remove('hidden');
}

function hideMetricsOverlay() {
    if (!metricsOverlay) return;
    metricsOverlay.classList.add('hidden');
}

function setStreaming(on) {
    isStreaming = on;
    updateSendButtonState();
    sendIcon.classList.toggle('hidden', on);
    stopIcon.classList.toggle('hidden', !on);
    sendBtn.classList.toggle('stop', on);
    queryInput.disabled = on;
}

/* ══════════════════════════════════════════════════════════
   Append User Message
══════════════════════════════════════════════════════════ */
function appendUserMsg(text) {
    showEmpty(false);
    const tpl = document.getElementById('msgUserTpl').content.cloneNode(true);
    tpl.querySelector('.msg-bubble-user').textContent = text;
    messages.appendChild(tpl);
    scrollToBottom();
}

/* ══════════════════════════════════════════════════════════
   Append AI Message (returns refs for live update)
══════════════════════════════════════════════════════════ */
function appendAiMsg() {
    const tpl = document.getElementById('msgAiTpl').content.cloneNode(true);
    const el = tpl.querySelector('.msg');
    const textEl = el.querySelector('.msg-text');
    const dotsEl = el.querySelector('.thinking-dots');
    const metaEl = el.querySelector('.msg-meta');
    const verdictEl = el.querySelector('.verdict');

    // New step indicator
    const stepIndEl = el.querySelector('.msg-step-indicator');
    const stepTextEl = el.querySelector('.msg-step-text');

    messages.appendChild(el);
    scrollToBottom();
    return { textEl, dotsEl, metaEl, verdictEl, el, stepIndEl, stepTextEl };
}

/* ══════════════════════════════════════════════════════════
   Tabs within message
══════════════════════════════════════════════════════════ */
function initTabs(metaEl) {
    const btns = metaEl.querySelectorAll('.tab-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            btns.forEach(b => b.classList.toggle('active', b === btn));
            metaEl.querySelectorAll('.tab-content').forEach(p => {
                p.classList.toggle('hidden', p.dataset.panel !== target);
            });
        });
    });
}

/* ══════════════════════════════════════════════════════════
   Render docs tab
══════════════════════════════════════════════════════════ */
function renderDocs(panel, docs) {
    panel.innerHTML = '';
    docs.forEach(doc => {
        const card = document.createElement('div');
        card.className = 'source-card';
        card.innerHTML = `
      <div class="source-card-header">
        <span class="source-rank">#${doc.rank}</span>
        <span class="source-file">${doc.source}</span>
        <span class="source-score">점수 ${doc.score.toFixed(3)}</span>
      </div>
      <div class="source-bar-wrap">
        <div class="source-bar-fill" style="width:${doc.pct}%"></div>
      </div>
      <div class="source-preview">${doc.preview}${doc.preview.length >= 280 ? '…' : ''}</div>`;
        panel.appendChild(card);
    });
}

/* ══════════════════════════════════════════════════════════
   Render perf tab
══════════════════════════════════════════════════════════ */
function renderPerf(panel, metrics, verifyResult, ragas) {
    const safe = (v, d = 0) => (Number.isFinite(Number(v)) ? Number(v) : d);
    const m = {
        search_s: safe(metrics?.search_s),
        rerank_s: safe(metrics?.rerank_s),
        gen_s: safe(metrics?.gen_s),
        verify_s: safe(metrics?.verify_s),
        total_s: safe(metrics?.total_s),
        ensemble_n: safe(metrics?.ensemble_n),
        final_n: safe(metrics?.final_n),
    };
    const externalDocsCount = safe(metrics?.external_docs_count);
    const fallbackTriggered = !!metrics?.external_fallback_triggered;
    const triggerReason = Array.isArray(metrics?.external_trigger_reason)
        ? metrics.external_trigger_reason.join(', ')
        : '';
    const extView = getExternalApiStatusView(metrics?.external_api_status);
    const f = (val) => (val || 0).toFixed(2);

    let ragasHtml = '';
    if (ragas) {
        ragasHtml = `
      <div style="font-size:.72rem;font-weight:600;color:var(--text-muted);margin-bottom:.4rem;text-transform:uppercase;letter-spacing:.05em">RAGAS 진단</div>
      <div class="ragas-detail-grid">
        <div class="metric-card-ragas">
          <div class="metric-label">신뢰성</div>
          <div class="metric-value">${f(ragas.faithfulness)}</div>
        </div>
        <div class="metric-card-ragas">
          <div class="metric-label">답변 관련성</div>
          <div class="metric-value">${f(ragas.answer_relevancy)}</div>
        </div>
      </div>`;
    }

    panel.innerHTML = `
    <div class="metric-total">
      <span class="metric-total-label">⏱️ 전체 소요 시간</span>
      <span class="metric-total-val">${m.total_s.toFixed(2)}s</span>
    </div>
    <div class="external-status-block">
      <div class="external-status-line ${extView.tone}">${escHtml(extView.text)}</div>
      <div class="external-status-sub">
        fallback: ${fallbackTriggered ? 'triggered' : 'not_triggered'} · external docs: ${externalDocsCount}${triggerReason ? ` · reason: ${escHtml(triggerReason)}` : ''}
      </div>
    </div>
    ${ragasHtml}
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">🔍 검색</div>
        <div class="metric-value">${m.search_s.toFixed(2)}s</div>
        <div class="metric-sub">${m.ensemble_n} → ${m.final_n}개</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">⚡ 리랭킹</div>
        <div class="metric-value">${m.rerank_s.toFixed(2)}s</div>
        <div class="metric-sub">CrossEncoder</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">✍️ 생성</div>
        <div class="metric-value">${m.gen_s.toFixed(2)}s</div>
        <div class="metric-sub">스트리밍</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">🧐 검증</div>
        <div class="metric-value">${m.verify_s.toFixed(2)}s</div>
        <div class="metric-sub">GPT Verifier</div>
      </div>
    </div>
    <div style="font-size:.72rem;font-weight:600;color:var(--text-muted);margin-bottom:.4rem;text-transform:uppercase;letter-spacing:.05em">검증 상세</div>
    <div class="verify-detail">${escHtml(verifyResult)}</div>`;
}

/* ══════════════════════════════════════════════════════════
   Render correction log tab
══════════════════════════════════════════════════════════ */
function renderLog(panel, logs, isPass) {
    if (!logs || logs.length === 0) {
        panel.innerHTML = `<div style="font-size:.82rem;color:var(--text-muted);padding:.5rem 0">
      ${isPass ? '✅ 자기 교정 없이 PASS 판정을 받았습니다.' : '교정 루프를 실행하지 않았습니다.'}
    </div>`;
        return;
    }
    panel.innerHTML = '';
    logs.forEach(log => {
        const rPass = log.verify_result.toUpperCase().includes('PASS');
        const div = document.createElement('div');
        div.className = 'correction-round';
        div.innerHTML = `
      <div class="correction-header" onclick="toggleRound(this)">
        <span class="correction-round-label ${rPass ? 'pass' : 'fail'}">
          ${log.round === 0 ? 'ROUND 0 (초기)' : 'ROUND ' + log.round}  ·  ${rPass ? 'PASS' : 'FAIL'}
        </span>
        <svg class="correction-chevron" width="14" height="14" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
      </div>
      <div class="correction-body">
        <div style="font-weight:600;margin-bottom:.3rem;font-size:.75rem;color:var(--text-secondary)">답변</div>
        <div style="margin-bottom:.6rem;font-size:.78rem;color:var(--text-secondary)">${renderMarkdown(log.answer)}</div>
        <div style="font-weight:600;margin-bottom:.3rem;font-size:.75rem;color:var(--text-secondary)">검증 결과</div>
        <div class="verify-detail">${escHtml(log.verify_result)}</div>
      </div>`;
        panel.appendChild(div);
    });

    // Auto-open last round
    const rounds = panel.querySelectorAll('.correction-round');
    if (rounds.length) {
        const last = rounds[rounds.length - 1];
        last.querySelector('.correction-body').classList.add('open');
        last.querySelector('.correction-chevron').classList.add('open');
    }
}

function toggleRound(header) {
    const body = header.nextElementSibling;
    const chev = header.querySelector('.correction-chevron');
    body.classList.toggle('open');
    chev.classList.toggle('open');
}

function escHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ══════════════════════════════════════════════════════════
   Clear Chat
══════════════════════════════════════════════════════════ */
function clearChat() {
    if (isStreaming && currentController) {
        currentController.abort();
    }
    hideMetricsOverlay();
    messages.innerHTML = '';
    showEmpty(true);
    clearStatus();
    setStreaming(false);
}

/* ══════════════════════════════════════════════════════════
   Send Message (SSE Streaming)
══════════════════════════════════════════════════════════ */
async function sendMessage() {
    if (isStreaming) {
        // Stop streaming
        if (currentController) currentController.abort();
        hideMetricsOverlay();
        setStreaming(false);
        clearStatus();
        return;
    }

    const query = queryInput.value.trim();
    if (!query) return;

    queryInput.value = '';
    autoResize();
    updateSendButtonState();

    appendUserMsg(query);
    const { textEl, dotsEl, metaEl, verdictEl, el, stepIndEl, stepTextEl } = appendAiMsg();
    hideMetricsOverlay();
    showMetricsOverlay('질문을 분석하고 답변을 준비 중이에요…');

    // Show thinking dots
    dotsEl.classList.remove('hidden');
    setStreaming(true);
    setStatus('🔍', '검색 중...');

    const controller = new AbortController();
    currentController = controller;

    const streamState = {
        tokenBuffer: '',
        textEl, dotsEl, metaEl, verdictEl, stepIndEl, stepTextEl,
        lastMetrics: null,
        lastVerifyResult: '',
        metricsPending: false,
    };

    try {
        const body = {
            query,
            model: modelSelect.value,
            top_k: parseInt(topkSlider.value),
            ensemble_k: 20,
            weight_bm25: parseInt(bm25Slider.value) / 100,
            use_self_correction: selfCorrEnabled,
        };

        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
        });

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buf += decoder.decode(value, { stream: true });

            // Parse SSE lines
            const lines = buf.split('\n');
            buf = lines.pop(); // keep incomplete line

            let eventType = '';
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    const raw = line.slice(6).trim();
                    try {
                        const payload = JSON.parse(raw);
                        handleSSE(eventType, payload, streamState);
                        // 상태 변화가 있을 때 VRAM 즉시 폴링
                        if (eventType === 'status' || eventType === 'done') pollVRAM();
                    } catch {/* skip bad JSON */ }
                    eventType = '';
                }
            }
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            // user stopped
        } else {
            dotsEl.classList.add('hidden');
            textEl.innerHTML = renderMarkdown(`❌ 오류가 발생했습니다: \`${err.message}\`\n\n서버가 실행 중인지 확인해 주세요.`);
            hideMetricsOverlay();
        }
    } finally {
        dotsEl.classList.add('hidden');
        if (!streamState.metricsPending) {
            hideMetricsOverlay();
        }
        setStreaming(false);
        if (!streamState.metricsPending) {
            clearStatus();
        }
        scrollToBottom();
        currentController = null;
        updateSendButtonState();
    }
}

/* ══════════════════════════════════════════════════════════
   SSE Event Handler
══════════════════════════════════════════════════════════ */
function handleSSE(type, payload, state) {
    const { textEl, dotsEl, metaEl, verdictEl, stepIndEl, stepTextEl } = state;
    switch (type) {
        case 'status':
            // Update global status text
            setStatus(payload.icon, payload.step);
            if (!state.tokenBuffer && !state.metricsPending) {
                showMetricsOverlay(payload.step);
            }
            // Hide thinking dots when specific status arrives
            if (dotsEl) dotsEl.classList.add('hidden');
            // Update in-bubble status
            if (stepIndEl && stepTextEl) {
                stepIndEl.classList.remove('hidden');
                stepTextEl.textContent = `${payload.icon} ${payload.step}`;
            }
            break;

        case 'token':
            // Hide progress indicators when tokens arrive
            if (stepIndEl) stepIndEl.classList.add('hidden');
            dotsEl.classList.add('hidden');
            hideMetricsOverlay();
            clearStatus();

            state.tokenBuffer += payload.text;
            textEl.innerHTML = renderMarkdown(state.tokenBuffer);
            scrollToBottom();
            break;

        case 'verdict':
            // 신규: 검증 결과 즉시 표시
            const isPassV = payload.is_pass;
            verdictEl.innerHTML = `
                <span class="verdict-badge ${isPassV ? 'pass' : 'fail'}">
                    ${isPassV ? '✅' : '⚠️'} 검증 ${isPassV ? 'PASS' : 'FAIL'}
                </span>`;
            break;

        case 'metrics_update':
            // 신규: RAGAS 지표 비동기 실시간 업데이트 (사이드바 + 활성 메시지 탭)
            if (payload) {
                const fVal = payload.faithfulness || 0;
                const rVal = payload.answer_relevancy || 0;

                // 1. 사이드바 업데이트
                if ($('valFaithfulness')) {
                    $('valFaithfulness').textContent = fVal.toFixed(2);
                    $('barFaithfulness').style.width = (fVal * 100) + '%';
                }
                if ($('valRelevancy')) {
                    $('valRelevancy').textContent = rVal.toFixed(2);
                    $('barRelevancy').style.width = (rVal * 100) + '%';
                }

                // 2. 현재 메시지 버블의 성능 탭 업데이트 (중요!)
                if (metaEl && !metaEl.classList.contains('hidden')) {
                    const perfPanel = metaEl.querySelector('[data-panel="perf"]');
                    if (perfPanel) {
                        renderPerf(
                            perfPanel,
                            state.lastMetrics || {},
                            state.lastVerifyResult || '',
                            payload,
                        );
                        console.log("Message Tab Metrics Updated");
                    }
                }
                if (state.metricsPending) {
                    state.metricsPending = false;
                    hideMetricsOverlay();
                    setStatus('✅', '성능 지표 업데이트 완료');
                    setTimeout(() => {
                        if (!isStreaming) clearStatus();
                    }, 1100);
                }
                console.log("Metrics Asynchronously Updated:", payload);
            }
            break;

        case 'done': {
            // Hide progress indicators when DONE
            if (stepIndEl) stepIndEl.classList.add('hidden');
            dotsEl.classList.add('hidden');
            clearStatus();

            // Final answer (might differ if self-correction ran)
            if (payload.answer) {
                textEl.innerHTML = renderMarkdown(payload.answer);
            }

            // Verdict badge (교정 로그 포함용 업데이트)
            const isPass = payload.is_pass;
            const rounds = payload.correction_rounds;
            const roundText = rounds > 0 ? ` <span class="verdict-rounds">(교정 ${rounds}회 후)</span>` : '';
            verdictEl.innerHTML = `
                <span class="verdict-badge ${isPass ? 'pass' : 'fail'}">
                    ${isPass ? '✅' : '⚠️'} 검증 ${isPass ? 'PASS' : 'FAIL'}${roundText}
                </span>`;

            // Build tabs
            metaEl.classList.remove('hidden');
            initTabs(metaEl);

            const docsPanel = metaEl.querySelector('[data-panel="docs"]');
            const perfPanel = metaEl.querySelector('[data-panel="perf"]');
            const logPanel = metaEl.querySelector('[data-panel="log"]');

            state.lastMetrics = payload.metrics || {};
            state.lastVerifyResult = payload.verify_result || '';
            state.metricsPending = !!payload.metrics_pending;
            if (state.lastMetrics?.external_api_status) {
                updateExternalApiStatus(state.lastMetrics.external_api_status);
            }

            renderDocs(docsPanel, payload.docs || []);
            renderPerf(perfPanel, state.lastMetrics, state.lastVerifyResult, payload.ragas);
            renderLog(logPanel, payload.correction_logs, isPass);

            if (state.metricsPending) {
                showMetricsOverlay('답변은 완료! 이제 RAGAS/성능 지표를 계산 중입니다…');
                setStatus('📊', '성능 지표 계산 중...');
            } else {
                hideMetricsOverlay();
            }

            // 최종 지표 동기화
            if (payload.ragas) {
                const fVal = payload.ragas.faithfulness || 0;
                const rVal = payload.ragas.answer_relevancy || 0;
                $('valFaithfulness').textContent = fVal.toFixed(2);
                $('barFaithfulness').style.width = (fVal * 100) + '%';
                $('valRelevancy').textContent = rVal.toFixed(2);
                $('barRelevancy').style.width = (rVal * 100) + '%';
            }

            // Update tab label
            const docTab = metaEl.querySelector('[data-tab="docs"]');
            if (docTab) docTab.textContent = `📄 참고 문서 (${(payload.docs || []).length})`;
            const logTab = metaEl.querySelector('[data-tab="log"]');
            if (logTab) logTab.textContent = `🔄 교정 로그 (${payload.correction_rounds || 0})`;

            scrollToBottom();
            break;
        }
        case 'error':
            if (dotsEl) dotsEl.classList.add('hidden');
            textEl.innerHTML = renderMarkdown(`❌ 서버 오류: \`${payload.message}\``);
            state.metricsPending = false;
            hideMetricsOverlay();
            break;
    }
}

/* ══════════════════════════════════════════════════════════
   Init
══════════════════════════════════════════════════════════ */
showEmpty(true);
queryInput.focus();
updateSendButtonState();
startPlaceholderRotation();
requestAnimationFrame(() => document.body.classList.add('ui-ready'));
