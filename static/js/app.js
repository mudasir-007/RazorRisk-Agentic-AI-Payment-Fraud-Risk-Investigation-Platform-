function escapeHtml(value) { const d=document.createElement('div'); d.textContent=String(value ?? ''); return d.innerHTML; }
// Report uncaught frontend errors into the server-side audit trail
// (logs/frontend_client.log) instead of a browser console only the person
// hitting the bug ever sees — best-effort, never throws itself.
function logClientError(message, context) {
    try {
        fetch(`${API_BASE}/api/v1/logs/client`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ level: 'error', message: String(message).slice(0, 500), context })
        }).catch(() => {});
    } catch (e) { /* logging must never itself break the page */ }
}
window.addEventListener('error', (e) => logClientError(e.message, 'window.onerror'));
window.addEventListener('unhandledrejection', (e) => logClientError(e.reason, 'unhandledrejection'));

// API_BASE is same-origin ("") when this dashboard is served by the FastAPI
// backend itself (Render, local dev). window.RAZORRISK_API_BASE is set in
// index.html — override it there when deploying static/ standalone (Vercel).
const API_BASE = window.RAZORRISK_API_BASE || '';

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadStats();
    loadRecentTransactions();
    loadAgentStatus();
    refreshLogStream();
    initVelocitySourceToggle();
    initInfoBubbles();

    // Form submit listener
    const form = document.getElementById('scoring-form');
    if (form) {
        form.addEventListener('submit', handleTransactionScore);
    }
});

// Tab Navigation
function initTabs() {
    const buttons = document.querySelectorAll('.nav-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));

            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(`tab-${tabId}`).classList.add('active');

            if (tabId === 'graph') {
                const userId = document.getElementById('user_id').value || 'USER_RING1_1';
                renderGraphTopology(userId);
            } else if (tabId === 'logs') {
                refreshLogStream();
            } else if (tabId === 'transactions') {
                loadRecentTransactions();
            } else if (tabId === 'agent') {
                loadAgentStatus();
            } else if (tabId === 'hitl') {
                loadHITLQueue();
            }
        });
    });
}

function updateVelocitySourceUI() {
    const toggle = document.getElementById('velocity_enabled');
    const label = document.getElementById('velocity-source-label');
    const wrap = document.getElementById('client-velocity-wrap');
    if (!toggle || !label || !wrap) return;
    const clientMode = toggle.checked;
    label.textContent = clientMode ? 'ON · Trust client-provided velocity' : 'OFF · Calculate from backend history';
    wrap.style.display = clientMode ? 'block' : 'none';
}

function initVelocitySourceToggle() {
    const toggle = document.getElementById('velocity_enabled');
    if (!toggle) return;
    toggle.addEventListener('change', updateVelocitySourceUI);
    updateVelocitySourceUI();
}

// Info bubbles (e.g. the velocity-source explainer) rely on CSS :hover for
// desktop, which doesn't exist on touch devices — this adds tap-to-toggle
// support so the same bubble works on mobile, plus tap-outside-to-close so
// it doesn't just stay pinned open.
function initInfoBubbles() {
    const triggers = document.querySelectorAll('.info-trigger');
    triggers.forEach((trigger) => {
        const bubble = trigger.nextElementSibling;
        if (!bubble || !bubble.classList.contains('info-bubble')) return;
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = bubble.classList.contains('is-open');
            document.querySelectorAll('.info-bubble.is-open').forEach((b) => b.classList.remove('is-open'));
            document.querySelectorAll('.info-trigger.is-active').forEach((t) => t.classList.remove('is-active'));
            if (!isOpen) {
                bubble.classList.add('is-open');
                trigger.classList.add('is-active');
            }
        });
    });
    document.addEventListener('click', () => {
        document.querySelectorAll('.info-bubble.is-open').forEach((b) => b.classList.remove('is-open'));
        document.querySelectorAll('.info-trigger.is-active').forEach((t) => t.classList.remove('is-active'));
    });
}

// Preset Scenario Handlers
function loadPreset(type) {
    // Every preset resets velocity_1h back to its default alongside
    // disabling the client-velocity toggle — previously only the toggle
    // was reset, so a value typed in while it was ON (including an invalid
    // one) silently survived a preset switch and could resurface later if
    // the toggle was re-enabled without the field being looked at again.
    document.getElementById('velocity_1h').value = '1';
    if (type === 'normal') {
        document.getElementById('user_id').value = 'USER_0042';
        document.getElementById('device_id').value = 'DEV_0088';
        document.getElementById('ip_address').value = '192.168.12.45';
        document.getElementById('amount').value = '1250';
        document.getElementById('merchant_id').value = 'MCH_005';
        document.getElementById('velocity_enabled').checked = false; updateVelocitySourceUI();
        document.getElementById('is_vpn_proxy').checked = false;
        document.getElementById('is_suspicious_proxy').checked = false;
    } else if (type === 'ring1') {
        document.getElementById('user_id').value = 'USER_RING1_1';
        document.getElementById('device_id').value = 'DEV_FRAUD_RING1';
        document.getElementById('ip_address').value = '185.220.101.44';
        document.getElementById('amount').value = '88000';
        document.getElementById('merchant_id').value = 'MCH_042';
        document.getElementById('velocity_enabled').checked = false; updateVelocitySourceUI();
        document.getElementById('is_vpn_proxy').checked = true;
        document.getElementById('is_suspicious_proxy').checked = true;
    } else if (type === 'ring2') {
        document.getElementById('user_id').value = 'USER_RING2_1';
        document.getElementById('device_id').value = 'DEV_RING2_USER_RING2_1';
        document.getElementById('ip_address').value = '198.51.100.99';
        document.getElementById('amount').value = '95000';
        document.getElementById('merchant_id').value = 'MCH_042';
        document.getElementById('velocity_enabled').checked = false; updateVelocitySourceUI();
        document.getElementById('is_vpn_proxy').checked = true;
        document.getElementById('is_suspicious_proxy').checked = true;
    } else if (type === 'carding') {
        document.getElementById('user_id').value = 'USER_CARDER_X';
        document.getElementById('device_id').value = 'DEV_CARDER_X';
        document.getElementById('ip_address').value = '203.0.113.50';
        document.getElementById('amount').value = '49';
        document.getElementById('merchant_id').value = 'MCH_012';
        document.getElementById('velocity_enabled').checked = false; updateVelocitySourceUI();
        document.getElementById('is_vpn_proxy').checked = true;
        document.getElementById('is_suspicious_proxy').checked = false;
    }
}

// Handle Transaction Scoring Submission
function handleTransactionScore(e) {
    e.preventDefault();
    const btn = document.getElementById('btn-score');
    btn.innerText = 'Scoring…';
    btn.disabled = true;

    const trustClientVelocity = document.getElementById('velocity_enabled').checked;
    const payload = {
        user_id: document.getElementById('user_id').value,
        device_id: document.getElementById('device_id').value,
        ip_address: document.getElementById('ip_address').value,
        amount: parseFloat(document.getElementById('amount').value),
        velocity_enabled: trustClientVelocity,
        merchant_id: document.getElementById('merchant_id').value,
        is_vpn_proxy: document.getElementById('is_vpn_proxy').checked,
        is_suspicious_proxy: document.getElementById('is_suspicious_proxy').checked
    };
    if (trustClientVelocity) {
        const rawVelocity = document.getElementById('velocity_1h').value;
        const parsedVelocity = parseInt(rawVelocity, 10);
        // parseInt('', 10) and parseInt('abc', 10) both return NaN, which
        // JSON.stringify silently turns into `null` — that null then fails
        // Pydantic's `ge=0` check server-side with no client-side warning
        // at all, so the request round-trips just to come back rejected.
        // Catching it here means a bad value never leaves the browser, and
        // the message is exact instead of "leaking" a backend error shape.
        if (Number.isNaN(parsedVelocity) || parsedVelocity < 0) {
            btn.innerText = 'Score Transaction & Run AI Engine';
            btn.disabled = false;
            alert('Client velocity must be a whole number of 0 or more.');
            return;
        }
        payload.velocity_1h = parsedVelocity;
    }

    fetch(`${API_BASE}/api/v1/transactions/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json().then(data => ({ ok: res.ok, status: res.status, data })))
    .then(({ ok, status, data }) => {
        if (!ok) {
            // The backend can reject this specific form on velocity_1h alone
            // (Pydantic's `ge=0` on the field, plus risk_aggregator.py's
            // "velocity_1h is required when velocity_enabled=true" check) —
            // every other field here is an unconstrained string/number, so
            // this is the one realistic way a validation error reaches this
            // handler. `data.detail` can be either a plain string (the
            // ValueError case) or FastAPI's list-of-objects shape (the
            // Pydantic case: [{type, loc, msg, input, ctx, url}, ...]) —
            // both used to fall straight through to `updateRiskDisplay`,
            // which crashed on the missing `risk_evaluation` key and threw
            // that raw structure into the user-facing alert. Extract a
            // clean, human-readable message here instead, and stop before
            // ever calling updateRiskDisplay with no evaluation to show.
            throw new Error(extractErrorMessage(data, status));
        }
        // Risk score renders the moment it's back — it no longer waits on
        // the (potentially several-seconds, LLM-backed) investigation step.
        btn.innerText = 'Score Transaction & Run AI Engine';
        btn.disabled = false;
        updateRiskDisplay(data.risk_evaluation);
        loadStats();
        loadRecentTransactions();
        loadHITLQueue();

        if (data.needs_investigation) {
            document.getElementById('agent-report-container').innerHTML = `
                <div class="placeholder-msg">Running agent investigation…</div>`;
            // Production path: enqueue into the shared Redis Streams queue so
            // the dashboard never blocks on ML/LLM investigation work.
            fetch(`${API_BASE}/api/v1/investigations/enqueue/${encodeURIComponent(data.transaction_id)}`, { method: 'POST' })
                .then(res => res.json().then(body => ({ ok: res.ok, status: res.status, body })))
                .then(({ ok, status, body }) => {
                    if (!ok) throw new Error(body.detail || `Investigation queue request failed (HTTP ${status})`);
                    // Local/no-Redis deployments (REDIS_REQUIRED=false) may return an
                    // already-terminal result inline instead of a job to poll for --
                    // see "degraded_mode" in api/routes_agent.py's enqueue fallback.
                    if (body.status === 'completed' || body.status === 'failed') return body;
                    return pollInvestigationJob(body.job_id);
                })
                .then(investigationRes => {
                    if (investigationRes.status === 'completed' && investigationRes.result) {
                        renderAgentReport(investigationRes.result);
                        return;
                    }
                    if (investigationRes.status === 'failed') {
                        throw new Error(investigationRes.error || 'Investigation worker failed.');
                    }
                    throw new Error(`Investigation ended in unexpected state: ${investigationRes.status}`);
                })
                .catch(err => {
                    console.error("Investigation error:", err);
                    document.getElementById('agent-report-container').innerHTML = `
                        <div class="placeholder-msg">Couldn't complete the investigation: ${escapeHtml(err.message)}</div>`;
                });
        } else {
            document.getElementById('agent-report-container').innerHTML = `
                <div class="placeholder-msg">
                    Risk score ${data.risk_evaluation.risk_score} is below the investigation threshold (70.0) — approved automatically, no agent report generated.
                </div>`;
        }
    })
    .catch(err => {
        btn.innerText = 'Score Transaction & Run AI Engine';
        btn.disabled = false;
        alert('Could not evaluate this transaction: ' + err.message);
    });
}

async function pollInvestigationJob(jobId, attempts = 0) {
    const maxAttempts = 120;
    const delayMs = 1000;
    const res = await fetch(`${API_BASE}/api/v1/investigations/jobs/${encodeURIComponent(jobId)}`);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || `Investigation status request failed (HTTP ${res.status})`);
    if (['completed', 'failed'].includes(body.status)) return body;
    if (attempts >= maxAttempts) throw new Error('Investigation status polling timed out in the dashboard.');
    await new Promise(resolve => setTimeout(resolve, delayMs));
    return pollInvestigationJob(jobId, attempts + 1);
}

// Turns a FastAPI/Pydantic error body into one readable line instead of
// letting its raw shape reach the user. `detail` is a plain string for a
// manually-raised backend error (e.g. risk_aggregator.py's ValueError), or
// a list of {loc, msg, ...} objects for a Pydantic field-validation error
// (e.g. velocity_1h failing `ge=0`) — both are handled explicitly here so
// neither ever "leaks" its raw structure into an alert box again.
function extractErrorMessage(data, status) {
    const detail = data && data.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && detail.length) {
        return detail.map(e => {
            const field = Array.isArray(e.loc) ? e.loc[e.loc.length - 1] : 'field';
            return `${field}: ${e.msg}`;
        }).join('; ');
    }
    return `Request failed (HTTP ${status})`;
}

function updateRiskDisplay(evalRes) {
    const score = evalRes.risk_score;
    const gauge = document.getElementById('risk-gauge');
    gauge.style.setProperty('--score-pct', `${score}%`);
    gauge.dataset.tier = (evalRes.risk_tier || 'LOW').toLowerCase();
    document.getElementById('score-val').innerText = score;

    const badge = document.getElementById('risk-tier-badge');
    badge.innerText = `${evalRes.risk_tier} RISK`;
    badge.className = `risk-badge tier-${(evalRes.risk_tier || 'low').toLowerCase()}`;

    const velocityState = `${evalRes.velocity_source || 'BACKEND'} · ${evalRes.velocity_1h ?? 0}/h`; 
    document.getElementById('decision-text').innerText = `Decision: ${evalRes.decision} · Velocity ${velocityState}${evalRes.review_id ? ' · HITL queued' : ''}`;

    // Update Progress Bars
    document.getElementById('val-tabular').innerText = `${evalRes.tabular_score}%`;
    document.getElementById('bar-tabular').style.width = `${evalRes.tabular_score}%`;

    document.getElementById('val-gnn').innerText = `${evalRes.gnn_score}%`;
    document.getElementById('bar-gnn').style.width = `${evalRes.gnn_score}%`;

    document.getElementById('val-topology').innerText = `${evalRes.stacker_calibrated_score}%`;
    document.getElementById('bar-topology').style.width = `${evalRes.stacker_calibrated_score}%`;

    // Evidence
    document.getElementById('ev-devices').innerText = `${evalRes.graph_evidence.shared_device_accounts} accounts`;
    document.getElementById('ev-ips').innerText = `${evalRes.graph_evidence.shared_ip_accounts} accounts`;
    document.getElementById('ev-community').innerText = `${evalRes.graph_evidence.community_size} users`;
}

function renderAgentReport(agentRes) {
    const container = document.getElementById('agent-report-container');
    // agentRes.summary_report is raw Markdown (agent/prompts.py REPORT_TEMPLATE).
    // marked.js converts it to HTML; if the CDN script didn't load for any
    // reason, fall back to plain text instead of throwing "marked is not defined".
    if (typeof marked !== 'undefined') {
        container.innerHTML = marked.parse(agentRes.summary_report);
    } else {
        console.error("marked.js failed to load — showing raw markdown as plain text.");
        container.innerText = agentRes.summary_report;
    }
    // Reflect the mode that ACTUALLY ran for this specific report — not
    // just the general "what would run next" status — so the badge never
    // implies more than the report it's sitting next to actually did.
    setAgentBadge(agentRes.agent_mode_label || agentRes.agent_mode, agentRes.agent_mode);

    // Reflect what actually ran for this report, not just the current toggle.
    if (agentRes.jev_verification) {
        const jv = agentRes.jev_verification;
        const label = jv.verification_flag === 'CONSISTENT'
            ? `Consistent (${Math.round(jv.independent_action_confidence * 100)}%)`
            : 'Review recommended';
        setJevBadge(label, jv.verification_flag);
    }
}

// Agent Mode Controls (Groq / Anthropic / OpenAI / deterministic-only)
function setAgentBadge(label, modeKey) {
    const badge = document.getElementById('agent-status-badge');
    if (!badge) return;
    badge.innerText = label;
    badge.className = 'agent-status-badge';
    if (modeKey && modeKey.indexOf('deterministic') !== -1) {
        badge.classList.add('status-deterministic');
    }
}

// Jev Verification Controls (TypeSafe System One post-investigation check)
function setJevBadge(label, flag) {
    const badge = document.getElementById('jev-status-badge');
    if (!badge) return;
    badge.innerText = label;
    badge.className = 'agent-status-badge';
    if (flag === 'REVIEW_RECOMMENDED') badge.classList.add('status-error');
    if (flag === 'off') badge.classList.add('status-deterministic');
}

function setJevMode(enabled) {
    const toggle = document.getElementById('jev-toggle');
    if (toggle) toggle.disabled = true;
    fetch(`${API_BASE}/api/v1/investigations/jev-mode`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
    }).then(res => res.json()).then(data => {
        setJevBadge(data.jev_verification_enabled ? 'On' : 'Off', data.jev_verification_enabled ? null : 'off');
    }).catch(err => {
        console.error("Jev mode change error:", err);
        alert("Couldn't change Jev verification mode: " + err.message);
        loadAgentStatus();
    }).finally(() => {
        if (toggle) toggle.disabled = !window.__jevConfigured;
    });
}

function loadAgentStatus() {
    fetch(`${API_BASE}/api/v1/investigations/agent-status`)
        .then(res => res.json())
        .then(data => {
            const select = document.getElementById('agent-mode-select');
            if (select) {
                select.innerHTML = '';
                data.modes.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.value;
                    opt.innerText = m.available ? m.label : `${m.label} (no API key)`;
                    opt.disabled = !m.available;
                    if (m.value === data.current_mode) opt.selected = true;
                    select.appendChild(opt);
                });
            }
            setAgentBadge(data.active_label, data.active_provider || 'deterministic');

            window.__jevConfigured = !!data.jev_configured;
            const jevToggle = document.getElementById('jev-toggle');
            if (jevToggle) {
                jevToggle.checked = !!data.jev_verification_enabled;
                jevToggle.disabled = !data.jev_configured;
            }
            setJevBadge(
                !data.jev_configured ? 'No API key' : (data.jev_verification_enabled ? 'On' : 'Off'),
                !data.jev_configured ? 'off' : null
            );
        })
        .catch(err => {
            console.error("Agent status fetch error:", err);
            setAgentBadge('Status unavailable', null);
            const badge = document.getElementById('agent-status-badge');
            if (badge) badge.classList.add('status-error');
        });
}

function setAgentMode(mode) {
    const select = document.getElementById('agent-mode-select');
    if (select) select.disabled = true;

    fetch(`${API_BASE}/api/v1/investigations/agent-mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
    })
        .then(async (res) => {
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Could not change agent mode');
            return data;
        })
        .then(data => {
            setAgentBadge(data.active_label, data.active_provider || 'deterministic');
        })
        .catch(err => {
            console.error("Agent mode change error:", err);
            alert("Couldn't change agent mode: " + err.message);
            loadAgentStatus(); // resync the dropdown with whatever the server actually has
        })
        .finally(() => {
            if (select) select.disabled = false;
        });
}

// Data Pipeline Controls (synthetic reseed + retrain)
function runDataPipeline(mode) {
    const statusEl = document.getElementById('pipeline-status');
    const synthBtn = document.getElementById('btn-seed-synthetic');
    const endpoint = `${API_BASE}/api/v1/admin/pipeline/synthetic`;
    const label = 'Regenerating synthetic data & retraining models…';

    statusEl.textContent = label;
    statusEl.className = 'pipeline-status status-loading';
    synthBtn.disabled = true;

    fetch(endpoint, { method: 'POST' })
        .then(async (res) => {
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Pipeline failed');
            return data;
        })
        .then(data => {
            const count = data.transactions_generated ?? data.transactions_ingested ?? 0;
            const stacked = data.eval_metrics && data.eval_metrics.stacked;
            const metricsStr = stacked ? ` — held-out ROC-AUC ${stacked.roc_auc?.toFixed(3)}, precision ${stacked.precision?.toFixed(2)}, recall ${stacked.recall?.toFixed(2)}` : '';
            statusEl.textContent = `Done — ${count.toLocaleString()} transactions (${data.mode}), models retrained${metricsStr}.`;
            statusEl.className = 'pipeline-status status-ok';
            loadStats();
            loadRecentTransactions();
        })
        .catch(err => {
            statusEl.textContent = `Couldn't finish — ${err.message}`;
            statusEl.className = 'pipeline-status status-error';
            console.error('Data pipeline error:', err);
        })
        .finally(() => {
            synthBtn.disabled = false;
            realBtn.disabled = false;
            // Clear a settled status message after a while so it doesn't sit
            // in the header permanently — but leave errors up longer, since
            // those need to actually be read and acted on.
            const lingerMs = statusEl.className.includes('status-error') ? 12000 : 6000;
            setTimeout(() => {
                if (statusEl.textContent) statusEl.textContent = '';
                statusEl.className = 'pipeline-status';
            }, lingerMs);
        });
}

function loadStats() {
    fetch(`${API_BASE}/api/v1/stats`)
        .then(res => res.json())
        .then(data => {
            document.getElementById('stat-total-txns').innerText = data.total_transactions;
            document.getElementById('stat-high-risk').innerText = data.high_risk_transactions;
            document.getElementById('stat-investigations').innerText = data.investigations_conducted;
        })
        .catch(err => console.error("Stats fetch error:", err));
}

function loadRecentTransactions() {
    fetch(`${API_BASE}/api/v1/transactions/recent?limit=15`)
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('txns-table-body');
            tbody.innerHTML = '';
            if (data.transactions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="12">No transactions logged yet.</td></tr>';
                return;
            }
            data.transactions.forEach(t => {
                const tr = document.createElement('tr');
                const tierClass = `tier-pill tier-${(t.risk_tier || 'unscored').toLowerCase()}`;
                const velocityLabel = `${t.velocity_source || 'BACKEND'} · ${t.velocity_1h ?? 0}/h`; 
                tr.innerHTML = `
                    <td><code>${escapeHtml(t.transaction_id)}</code></td>
                    <td><code>${escapeHtml(t.user_id)}</code></td>
                    <td>${escapeHtml(t.device_id)}</td>
                    <td>${escapeHtml(t.ip_address)}</td>
                    <td>₹${Number(t.amount).toLocaleString()}</td>
                    <td><strong>${velocityLabel}</strong></td>
                    <td>${Number(t.gnn_score).toFixed(1)}%</td>
                    <td>${Number(t.tabular_score).toFixed(1)}%</td>
                    <td>${Number(t.stacker_calibrated_score).toFixed(1)}%</td>
                    <td><strong>${Number(t.risk_score).toFixed(1)}</strong></td>
                    <td><span class="${tierClass}">${escapeHtml(t.risk_tier)}</span></td>
                    <td><code>${escapeHtml(t.decision)}</code></td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => console.error("Error loading txns:", err));
}

function refreshLogStream() {
    fetch(`${API_BASE}/api/v1/logs/stream?lines=25`)
        .then(res => res.json())
        .then(data => {
            document.getElementById('log-risk-engine').innerText = data.risk_engine_logs.join('\n');
            document.getElementById('log-agent').innerText = data.agent_logs.join('\n');
        })
        .catch(err => console.error("Error streaming logs:", err));
}


async function loadHITLQueue() {
    const box = document.getElementById("hitl-queue");
    if (!box) return;
    try {
        const res = await fetch(`${API_BASE}/api/v1/hitl/queue`);
        const data = await res.json();
        if (!data.reviews || data.reviews.length === 0) {
            box.innerHTML = '<p class="placeholder-msg">No pending human reviews.</p>';
            return;
        }
        box.innerHTML = data.reviews.map(r => `
            <div class="card" style="margin-bottom:12px;">
                <strong>${escapeHtml(r.transaction_id)}</strong>
                <span>Risk ${Number(r.risk_score).toFixed(1)}</span>
                <p><strong>Reasons:</strong> ${escapeHtml((r.reasons || []).join(", ") || "Policy escalation")}</p>
                <p><strong>Status:</strong> ${escapeHtml(r.status)} · Queued ${escapeHtml(r.created_at || "")}</p>
                <pre>${escapeHtml(JSON.stringify(r.evidence || {}, null, 2))}</pre>
                <button class="btn-preset" onclick="resolveHITL('${r.review_id}','APPROVE')">Approve</button>
                <button class="btn-preset" onclick="resolveHITL('${r.review_id}','HOLD')">Hold</button>
                <button class="btn-preset tier-tag-critical" onclick="resolveHITL('${r.review_id}','BLOCK')">Block</button>
            </div>`).join("");
    } catch (e) {
        box.innerHTML = `<p class="placeholder-msg">Could not load review queue.</p>`;
    }
}

async function resolveHITL(reviewId, decision) {
    const rationale = window.prompt("Human rationale (required):");
    if (!rationale) return;
    await fetch(`${API_BASE}/api/v1/hitl/review/${encodeURIComponent(reviewId)}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({decision, reviewer: "dashboard-reviewer", rationale})
    });
    await loadHITLQueue();
}
