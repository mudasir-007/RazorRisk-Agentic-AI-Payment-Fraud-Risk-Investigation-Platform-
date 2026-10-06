# 🗺️ RazorRisk — Complete Project Architecture & Codebase Workflow Guide

This document is a comprehensive, step-by-step technical guide to understanding the **RazorRisk** codebase from
scratch: how every component works, how data flows through the system, how models are trained leak-free and
combined by a learned stacker, and how the investigation agent produces an audit-ready report whether or not an
LLM is configured.

---

## Workflow Diagram

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart TB
    T[Incoming transaction] --> F[Feature engineering]
    F --> V[Velocity]
    F --> TM[Tabular ML]
    F --> G[User-only graph]
    G --> GN[GraphSAGE]
    TM --> ST[Learned stacker]
    GN --> ST
    ST --> P[Risk aggregation + policy]
    V --> P
    P --> M{Mandatory-human reason?<br/>uncertainty / disagreement /<br/>evidence conflict / high-impact}
    M -->|Yes, always| H[HITL]
    M -->|No| C{Confidence >= 0.95?}
    C -->|Yes| AB[Auto-block, no human]
    C -->|No| A[Approve / Monitor / Hold<br/>by risk tier]
    H --> I[Investigation + reviewer]
    A --> AU[Audit]
    AB --> AU
    I --> AU
```

A confidence threshold on its own never bypasses the mandatory-human reasons — model disagreement, evidence conflict, model uncertainty, and high-dollar-amount (`HIGH_IMPACT`) transactions always route to `H[HITL]` regardless of how confident the stacker is. Only an unambiguous, high-confidence score with none of those reasons present takes the `AB[Auto-block]` path. See Bug #27 below and [`ml/decision_policy.py`](ml/decision_policy.py).

### Training and evaluation data flow

The current model contract uses **one expanded synthetic RazorRisk population** for all three ML components. XGBoost consumes transaction-level features; GraphSAGE consumes the canonical User-only graph; the stacker consumes paired predictions for the same held-out transactions. This preserves the complementary-view design without mixing incompatible external datasets.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 28, 'rankSpacing': 50}}}%%
flowchart TB
    D[Expanded synthetic generator] --> S[User-level split]
    S --> T[Tabular features]
    S --> G[User-only graph]
    S --> L[Fraud labels]
    T --> X[XGBoost]
    G --> GN[GraphSAGE GNN]
    X --> XO[Held-out XGB scores]
    GN --> GO[Held-out GNN scores]
    XO --> P[Paired predictions]
    GO --> P
    L --> P
    P --> ST[Balanced logistic stacker]
    ST --> E[ROC-AUC / PR-AUC / precision / recall / F1]
```

### Live dual-model inference

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 28, 'rankSpacing': 50}}}%%
flowchart LR
    TX[Incoming transaction] --> API[FastAPI score endpoint]
    API --> TF[Transaction feature builder]
    API --> UG[Current User-only graph]
    TF --> X[XGBoost]
    UG --> G[GraphSAGE]
    X --> XP[XGB probability]
    G --> GP[GNN probability]
    XP --> ST[Learned stacker]
    GP --> ST
    ST --> R[Combined model probability]
    R --> P[Policy + security guardrails]
    P --> H{HITL required?}
    H -->|Yes| Q[Pending human review]
    H -->|No| A[Automatic action by policy]
    Q --> I[Investigation / reviewer]
    A --> AU[Audit]
    I --> AU
```

## 1. Mental Model & Core Architecture Shift

There are **two graphs** in this codebase, deliberately kept separate:

```mermaid
flowchart LR
    subgraph RG["Canonical risk graph (ml/risk_graph.py) — User nodes ONLY"]
        U1((User)) -->|"shares device (w=2)"| U2((User))
        U1 -->|"shares IP (w=1)"| U3((User))
    end

    subgraph VG["Dashboard visualization graph (ml/graph_builder.py) — richer, for humans"]
        VU((User)) -->|USES_DEVICE| VD["Device"]
        VU -->|USES_IP| VI{"IP"}
        VU -->|TRANSACTS_WITH| VM["Merchant"]
    end
```
The risk graph drives GNN training + Louvain community detection — a fraud ring is a dense User-only cluster.
The visualization graph is only ever read by the Graph Topology tab, never fed into any model.

These used to be the same graph, which caused a real bug: 2-hop traversal through a popular Merchant node (used
by hundreds of unrelated users) made a 7-person fraud ring balloon into a 692-node unreadable subgraph. A
Merchant used by thousands of unrelated users was never real *fraud* signal - it just happened to be reachable
in 2 hops - so the graph that actually trains the model never includes Merchant/heavy-fanout nodes as hops at
all. The dashboard's separate graph keeps its own traversal cap for its own different job (human-readable
exploration).

Traditional payment fraud engines evaluate each transaction independently: Transaction -> Features -> Tabular
ML Model -> Fraud Probability.

**RazorRisk** combines that with a graph-aware model and lets a *learned* stacker decide how much to trust each:

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 32, 'rankSpacing': 55}}}%%
flowchart TB
    TXN["Incoming Payment"] --> GW["FastAPI /transactions/score"]
    GW --> VEL["Server-side 1h velocity count"]
    VEL --> TAB["Tabular ML / XGBoost"]
    GW --> GNN["User-only Risk Graph + GraphSAGE"]
    TAB --> STACK["Learned Logistic Stacker"]
    GNN --> STACK
    STACK --> CAL["Stacker calibrated probability"]
    CAL --> MULT["Velocity / proxy overlay<br/>(client-trust ON / backend-calc OFF)"]
    MULT --> SCORE["Composite risk 0-100"]
    SCORE --> POLICY["Decision Policy + guardrails"]
    POLICY --> MANDH{"Mandatory-human reason?<br/>uncertainty / disagreement /<br/>evidence conflict / high-impact"}
    MANDH -->|"Yes, always"| QUEUE["human_reviews: PENDING"]
    MANDH -->|"No"| CONF{"Calibrated confidence >= 0.95?"}
    CONF -->|"Yes"| AUTOBLOCK["Auto-block<br/>no human in loop"]
    CONF -->|"No"| DECIDE["APPROVE / MONITOR / HOLD /<br/>BLOCK_PENDING_REVIEW by tier"]
    QUEUE --> REVIEW["Human reviewer"]
    SCORE --> PERSIST["transactions + risk_scores"]
    PERSIST --> QUEUE
    SCORE --> INV{"High risk or HITL?"}
    INV -->|"yes, separate call"| AGENT["Investigation Agent"]
    AGENT --> REPORT["Investigation report"]
```


---

## 2. Recommended Reading Order (Where to Start in Code)

```text
1. System Configuration & Logger
   - config.py
   - utils/logger.py             8 subsystem channels + correlation IDs

2. Database Schema & Data Models
   - db/schema.sql
   - db/database.py             PostgreSQL-first connection layer; SQLite is explicit test/local fallback

3. Synthetic Data & Fraud Scenarios
   - data/generate_synthetic_data.py    each user has ONE dedicated device/IP + benign co-location noise
   - data/ingest_real_kaggle_dataset.py legacy external-dataset experiment; not used by the current model/evaluation contract

4. Shared ML utilities & canonical risk graph
   - ml/common.py                user-level train/test split, shared by both models
   - ml/risk_graph.py            User-only weighted graph + Louvain communities

5. The two independently-trained models
   - ml/train_tabular_model.py   XGBoost (sklearn fallback), leak-free SQL window features
   - ml/train_gnn.py             hand-rolled NumPy GraphSAGE, inductive inference

6. The learned combination
   - ml/risk_aggregator.py       train_stacker() (offline) + calculate_composite_risk_score() (live)

7. Investigation agent
   - agent/tools.py              4 deterministic evidence tools, identical for both modes
   - agent/llm_investigator.py   real LLM path (Anthropic/Groq/OpenAI), only if a key is set
   - agent/deterministic_agent.py  honestly-labeled rule-based fallback
   - agent/prompts.py
   - agent/graph_agent.py        dispatcher between the two

8. FastAPI REST API Gateway & Logging API
   - api/main.py
   - api/routes_transactions.py  binds a correlation ID per scored transaction
   - api/routes_graph.py
   - api/routes_agent.py
   - api/routes_logs.py          includes POST /logs/client for frontend error capture
   - api/routes_admin.py         reseed/ingest + full retrain, one call

9. Frontend Dashboard & Vis.js Visualizer
   - static/index.html
   - static/css/styles.css       tier-driven risk color, not "everything is red"
   - static/js/app.js
   - static/js/graph_vis.js

10. Tests
    - tests/test_risk_engine.py  full pipeline integration test, runs offline
```

---

## 3. End-to-End Execution Sequence (Step-by-Step Flow)

```mermaid
sequenceDiagram
    participant C as Client
    participant API as Transactions API
    participant TAB as Tabular ML
    participant GNN as GraphSAGE
    participant AGG as Stacker / Risk Aggregator
    participant POL as Decision Policy
    participant DB as SQLite
    participant HITL as Human Review Queue
    participant AGT as Investigation Agent
    participant LOG as Audit Logs

    C->>API: POST /transactions/score
    API->>AGG: score transaction
    AGG->>DB: count prior txns in trailing 1h
    AGG->>TAB: score behavioral features
    AGG->>GNN: score user graph node
    TAB-->>AGG: Tabular ML score
    GNN-->>AGG: GNN node-embedding score
    AGG->>AGG: learned stacker -> calibrated score
    AGG->>AGG: optional velocity/proxy overlay
    AGG->>POL: composite risk + evidence
    POL->>POL: check mandatory-human reasons<br/>(uncertainty / disagreement / conflict / high-impact)
    alt mandatory-human reason present
        POL-->>API: HUMAN_REVIEW, hitl_required=true
    else confidence >= 0.95 and no mandatory reason
        POL-->>API: BLOCK (auto), hitl_required=false
    else
        POL-->>API: tier-based decision, hitl_required=false
    end
    API->>DB: persist transaction + all 3 model scores
    alt HITL required
        API->>HITL: enqueue PENDING review after DB commit
        HITL-->>C: review_id returned
    end
    API-->>C: risk evaluation + correlation_id
    API->>LOG: model scores + velocity + decision
    opt investigation requested
        C->>API: POST /investigations/run/{txn_id}
        API->>AGT: gather evidence + investigate
        AGT->>DB: persist investigation report
        AGT-->>C: investigation report
    end
```


### Step 1: Request Ingestion (`api/routes_transactions.py`)
`POST /api/v1/transactions/score` receives a transaction payload and immediately calls
`bind_correlation_id()` — every log line emitted anywhere in the next 4 steps carries this same ID.

### Step 2: Velocity Source Selection and Tabular ML Fraud Scoring (`ml/risk_aggregator.py`)
- Selects hourly velocity from the explicit frontend source toggle: client-provided in trust mode, or server-computed from the trailing-1h transaction history in backend mode. The selected value then feeds the same model features used at training time (`amount_log`, `hour_of_day`, `day_of_week`,
  `velocity_1h`, `amount_zscore_prior`, `merchant_fraud_rate`) from live DB queries — `amount_zscore_prior` and
  `merchant_fraud_rate` both use ONLY information available before this transaction (prior transactions' mean/
  std, and merchant rates fit only on the training split).
- Calls `ml/train_tabular_model.py::predict_tabular_fraud_prob()`, which loads the trained XGBoost (or sklearn
  fallback) model.

### Step 3: Graph & GNN Node Scoring (`ml/risk_aggregator.py::live_gnn_score_and_evidence`)
- Rebuilds the canonical User-only risk graph (`ml/risk_graph.py::build_user_graph`) from current DB contents when the live snapshot is missing/invalidated or its TTL expires. The API invalidates the snapshot after every committed transaction so rapid repeated payments cannot score against stale device/IP/community topology.
- Runs one **inductive** forward pass (`GraphSAGEInference.score_all`) with the already-trained weights over
  whatever the current graph looks like — any user present gets a real score, including ones added after
  training, no cache and no fallback heuristic needed.
- Also returns graph evidence (shared device/IP account counts, community size) for the dashboard's "Instant
  Graph Evidence" panel and the agent's `GraphTool`.

### Step 4: Composite Risk Scoring (`ml/risk_aggregator.py::calculate_composite_risk_score`)
- Combines `tabular_score` and `gnn_score` via the **learned logistic-regression stacker**
  (`coef[0]*tabular + coef[1]*gnn + intercept`, sigmoid) — coefficients are fit offline by `train_stacker()` on
  held-out data, not hand-picked.
- Applies the velocity/VPN-proxy multiplier **after** that calibrated probability, as an explicit rule-based
  overlay — logged and returned separately (`velocity_multiplier`), never folded into the "learned" score.
- Classifies Risk Tier & Decision:
  - `0 – 39.9`: `LOW` → `APPROVE`
  - `40.0 – 69.9`: `MEDIUM` → `MONITOR`
  - `70.0 – 89.9`: `HIGH` → `HOLD_FOR_INVESTIGATION`
  - `90.0 – 100.0`: `CRITICAL` → `BLOCK_AND_INVESTIGATE`

### Step 5: Policy, HITL Queue, and Separate Investigation Dispatch (`agent/graph_agent.py`)
- If Risk Score ≥ 70.0 **or HITL policy requires review**, `/api/v1/transactions/score` returns `needs_investigation: true` **without** running the
  investigation itself — the risk score, tier, and evidence render in the dashboard immediately. The frontend
  (`static/js/app.js`) then fires a separate `POST /api/v1/investigations/run/{transaction_id}` and fills in the
  report panel once that resolves. This split exists because the investigation step is materially slower than
  the risk scoring step (an LLM call can take several seconds; risk scoring is sub-second) — an earlier version
  ran both synchronously inside `/score`, so the whole UI waited on the slowest part even when nothing about the
  fast part had anything left to compute.
- **Evidence gathering is identical regardless of mode** — 4 deterministic tools (`agent/tools.py`):
  `GraphTool`, `TransactionHistoryTool`, `DeviceRiskTool`, `FraudModelTool`.
- **Hypothesis generation** then tries `agent/llm_investigator.py` if `ANTHROPIC_API_KEY` / `GROQ_API_KEY` /
  `OPENAI_API_KEY` is set; on any failure (or no key at all) falls back to `agent/deterministic_agent.py`'s
  rule-based pattern matching. The resulting report's `agent_mode` field states which one actually ran.
- Which path runs can also be forced from the dashboard's **Agent mode** selector — `GET /agent-status` /
  `POST /agent-mode` in `api/routes_agent.py`, backed by `agent/mode_state.py`'s process-local in-memory
  override (`auto` / a specific provider / `deterministic`). Useful for demoing "this is what the LLM path
  looks like" vs. "this is the fallback" on demand, without needing different API keys configured.

### Step 6: Database Persistence & Log Streaming (`db/database.py` & `utils/logger.py`)
- Saves transaction, risk score, and investigation report to SQLite (`db/database.py`). The application uses one raw-SQLite data path.
- Logs route to 8 subsystem channels: `app`, `risk_engine`, `agent`, `ml_training`, `graph`, `database`,
  `pipeline`, `frontend_client` — every line for this request carries the correlation ID from Step 1.
- The live in-memory dashboard graph gets this one transaction folded in incrementally
  (`graph_builder.add_transaction`) — no full rebuild needed for a single live score.

### Step 6b: Human-in-the-Loop Queue (`api/routes_hitl.py`)
- `apply_decision_policy()` can trigger `MODEL_UNCERTAINTY`, `MODEL_DISAGREEMENT`, `HIGH_IMPACT`, `EVIDENCE_CONFLICT`, or `NOVEL_BEHAVIOR`.
- The API commits the transaction and risk record first, then creates an idempotent `PENDING` row in `human_reviews`.
- The dashboard refreshes the queue after every score and exposes `APPROVE`, `HOLD`, and `BLOCK` reviewer actions.
- A review is not an external email/task-system integration; it is an internal RazorRisk reviewer queue backed by SQLite.

### Step 7: Dashboard UI Update (`static/js/app.js` & `static/js/graph_vis.js`)
- Updates the risk gauge (color driven by actual tier, not hardcoded red) and the three model-breakdown bars —
  the third bar shows the stacker's calibrated score, not a fabricated "topology risk" number. This happens the
  instant `/score` responds, independent of whether an investigation is still running.
- Vis.js renders a capped, fraud-signal-focused 2-hop network around the user (Device/IP hops only — Merchant
  fan-out is excluded, see §1).
- Displays the agent report once its separate follow-up call resolves (states its own mode), parsed from
  Markdown to HTML via `marked.js` rather than dumped as raw text, and streams log lines into the audit console.
- Any uncaught frontend JS error gets POSTed to `/api/v1/logs/client` automatically.
- All fetch calls go through an `API_BASE` constant (`window.RAZORRISK_API_BASE`, set in `index.html`) instead
  of hardcoded relative paths — empty by default (same-origin), overridable when the dashboard is deployed
  standalone against a different backend origin. See §4 below.

---

## 4. Deployment & Ops

Deployment covers three platforms (Render, Vercel, Hugging Face Spaces) plus a distributed
PostgreSQL+Redis-backed runtime for horizontal scaling — see `README.md`'s **Deployment**,
**Production Data Plane**, and **Production Distributed Runtime** sections for step-by-step instructions
and architecture.

Bugs #1–13 (foundational architecture + the deployment/multi-platform fixes above) and #31–36
(PostgreSQL migration, Redis-backed rate limiting, and their local-dev fallback) are documented in full,
with canonical numbering, in **[BUGS.md](BUGS.md)**.

## 4.5 Bugs & Regression History

Bugs #14–30 (found during manual and automated validation, after the architecture already looked "done")
are documented in full in **[BUGS.md § Phase 3](BUGS.md#phase-3--testing--regression-validation-bugs-1430)**,
including two disclosed, still-open detection gaps from Bug #29 and the Bug #30 new-user auto-block fix.

## 4.6 Jev Verification Layer

Jev is an optional post-investigation verification layer. It does not participate in XGBoost/GNN scoring, stacker calibration, or the primary risk decision.

The execution order is:

1. deterministic evidence tools gather graph, history, device, and model evidence;
2. the configured LLM investigator or deterministic fallback produces the hypothesis and recommended action;
3. if the Jev toggle is enabled and `TYPESAFE_API_KEY` is configured, Jev independently checks the recommended action and checks whether the hypothesis is grounded in the evidence;
4. `CONSISTENT` requires action agreement and grounding above the configured threshold;
5. a pending HITL review may be auto-resolved only when Jev confidence clears `JEV_AUTO_RESOLVE_MIN_CONFIDENCE` and no `MANDATORY_HUMAN_REASONS` apply;
6. disagreement, weak grounding, unavailable Jev, malformed responses, or mandatory-human reasons leave the human-review path intact.

The implementation is intentionally fail-open for report generation: a Jev dependency failure is logged and does not prevent the underlying investigation report from being returned.

## 5. Deep-Dive Component Map

### A. Database Layer (`db/`)
- `schema.sql` — `users`, `devices`, `ip_addresses`, `merchants`, `transactions`, `risk_scores`,
  `investigation_reports`, `human_reviews`, `system_logs`.
- `database.py` — raw `sqlite3` connection helper + schema init. One application-owned SQLite data path.

### B. Synthetic Data & Fraud Ring Generator (`data/generate_synthetic_data.py`)
Generates ~12,000 transactions. Each normal user gets **their own dedicated device + IP** (a person doesn't
switch phones between purchases) plus ~60 benign IP co-locations (pairs of unrelated users who happen to share
one IP — roommates, office wifi — with no elevated fraud behavior) so "any shared IP" isn't a perfect signal by
construction. Injects 4 explicit fraud scenarios: device-sharing ring, IP/proxy botnet, carding micro-
transactions, and merchant collusion.

### C. Machine Learning Engine (`ml/`)
- `common.py` — `user_level_split()`, shared by both models so a fraud-ring member's transactions never appear
  on both sides of train/test.
- `risk_graph.py` — canonical User-only weighted graph + Louvain communities + node feature extraction +
  row-normalized adjacency for the GNN.
- `train_tabular_model.py` — XGBoost (sklearn `HistGradientBoostingClassifier` fallback), leak-free SQL window-
  function features, merchant target-encoding fit only on train, real held-out `classification_report_dict`.
- `train_gnn.py` — 2-layer GraphSAGE mean-aggregation network, implemented from scratch in NumPy: each layer
  concatenates a node's own features with the mean of its neighbors' features, then applies a linear layer and
  ReLU. Trains with manual backprop, saves weights to `ml/models/gnn_model.npz`. `GraphSAGEInference`
  re-implements just the forward pass for live, inductive scoring.
- `risk_aggregator.py` — `train_stacker()` (offline logistic regression fit) + `calculate_composite_risk_score()`
  (live scoring entrypoint, the API's hot path).
- `graph_builder.py` — the separate dashboard visualization graph (User/Device/IP/Merchant), untouched by
  training.

### D. Investigation Agent (`agent/`)
- `tools.py` — `GraphTool`, `TransactionHistoryTool`, `DeviceRiskTool`, `FraudModelTool`: deterministic,
  identical for both agent modes.
- `llm_investigator.py` — real LLM call (LangChain chat-model interface; Anthropic → Groq → OpenAI, first
  configured key wins), only reasons over evidence already gathered, never computes a number itself.
- `deterministic_agent.py` — rule-based hypothesis/action selection, the fallback path.
- `prompts.py` — the markdown report template, shared by both modes.
- `graph_agent.py` — `RiskInvestigationAgent.investigate()`: gathers evidence, tries LLM, falls back, renders
  the report, labels `agent_mode` honestly.

### E. FastAPI REST Gateway & Logs (`api/`)
- `main.py` — app init, builds the dashboard graph at startup (so live scoring has graph signal from process
  start, not only after some other code path happens to touch it).
- `routes_transactions.py` — `/api/v1/transactions/score`, `/api/v1/transactions/recent`; binds/clears the
  correlation ID per request.
- `routes_graph.py` — `/api/v1/graph/topology/{user_id}` (capped, Device/IP-only 2nd hop),
  `/api/v1/graph/communities`.
- `routes_agent.py` — `/api/v1/investigations/run/{txn_id}`, `/api/v1/investigations/{txn_id}`, plus
  `GET /agent-status` / `POST /agent-mode` for the dashboard's live agent-mode selector (backed by
  `agent/mode_state.py`'s in-memory override — resets to `auto` on restart, not persisted app config).
- `routes_logs.py` — `/api/v1/logs/stream` (all 8 channels), `/api/v1/logs/client` (frontend error capture).
- `routes_admin.py` — `/api/v1/admin/pipeline/synthetic`: regenerate synthetic data, rebuild
  the dashboard graph, run the full tabular → GNN → stacker retrain, return held-out eval metrics, and drop the
  live-scoring process's cached model weights (`_LiveModels.reset()`) so the next request picks up the fresh
  ones. `/api/v1/admin/rebuild-graph` refreshes just the dashboard graph without retraining.

---

## 6. How to Explain This Project in an Interview

1. **The Core Problem**:
   > *"Existing payment fraud models look at transactions in isolation. If 7 stolen credit cards are used on 7 new accounts, a row-level model evaluates 7 independent 'normal-looking' transactions. RazorRisk builds a User-only weighted graph from shared device/IP fingerprints and runs community detection on it, so 7 accounts sharing one device show up as a dense cluster regardless of how normal each individual transaction looks."*

2. **The Dual ML Architecture, Combined by a Learned Stacker — Not a Guess**:
   > *"I train two independent models on the same leak-free, user-level train/test split: an XGBoost model on transaction-level behavioral features, and a GraphSAGE GNN — implemented from scratch in NumPy, since at this scale a PyTorch dependency wasn't justified — on the user-risk graph. Rather than hand-picking how to weight them, I fit a small logistic regression stacker on their held-out scores, and I log both the learned coefficients and a tabular-only vs GNN-only vs stacked comparison every retrain, so I can show the combination is actually additive."*

3. **The Agentic Layer, Honestly Scoped**:
   > *"When a transaction scores ≥ 70, an investigation agent gathers evidence through four deterministic tools — shared-entity graph queries, transaction history, device risk, model scores. If an LLM API key is configured, a real model reasons over that evidence to write the hypothesis and recommended action; if not, a rule-based fallback does the same job. Either way, the agent can't fabricate a number — every figure in the report traces back to one of the four tools — and the report states plainly which mode produced it."*

4. **Production Readiness & Auditability**:
   > *"Every subsystem — API, risk engine, agent, ML training, graph, database, pipeline, frontend — logs to its own rotating file, and every scored transaction gets a correlation ID that ties its log lines together across all of them. I can hand someone a transaction ID and they can grep the exact decision trail across every model that touched it."*

5. **The Deployment Split — a Real Constraint, Not a Preference**:
   > *"XGBoost, SciPy, and the LangChain provider packages together install to about 2GB, and Vercel's serverless Python functions cap out around 250MB unzipped — so the backend was never going to run there as a function no matter how I configured it. I run the backend on Render, a real persistent process, and optionally mirror just the static dashboard on Vercel pointed at the Render URL over CORS. That decision also surfaced a real bug: the raw SQLite connection helper had a hardcoded path that happened to agree with the SQLAlchemy engine's path locally and on Render, but would have silently diverged — and crashed every write — the moment I tried running any part of this somewhere with a read-only filesystem. Now there's one source of truth for that path instead of two that happened to match."*

---

## Evaluation Reproduction Contract

The checked-in evaluation runner is the source of truth for README model metrics:

```bash
python tests/evaluate_models.py --dataset synthetic
python tests/evaluate_models.py --dataset synthetic --retrain
python tests/evaluate_models.py --dataset synthetic
```

Synthetic evaluation compares XGBoost, the user-level GraphSAGE score, and the learned stacker on the same held-out synthetic transaction population. The stacker is explicitly class-balanced and consumes paired predictions from those same transactions. The ULB/Kaggle dataset is outside the RazorRisk model/evaluation contract because its anonymized PCA feature space and missing identity relationships do not match the project domain.

## Hyperparameter Selection Workflow

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 50}}}%%
flowchart TB
    A["Synthetic Users + Transactions"] --> B["User-level CV folds"]
    B --> C["XGBoost candidates"]
    B --> D["GNN candidates"]
    C --> E["CV PR-AUC"]
    D --> E
    E --> F["Best base configurations"]
    F --> G["OOF base predictions"]
    G --> H["Balanced stacker candidates"]
    H --> I["CV PR-AUC"]
    I --> J["Best stacker"]
    J --> M["ml/models/hyperparameters.json"]
    M --> K["Final retraining<br/>(train_tabular_model / train_gnn / train_stacker,<br/>via ml/common.py::load_tuned_hyperparameters — Bug #28)"]
    K --> L["Untouched test set"]
```

### Anti-leakage rules

- User-level folds keep all transactions for a user in one fold.
- Merchant target encoding is fitted only on the training portion of each XGBoost fold.
- XGBoost class weighting is computed from each fold's training labels.
- GNN labels are used only for the loss; graph construction does not use ground-truth labels.
- Stacker tuning uses out-of-fold base predictions rather than in-sample base predictions.
- Stacker candidates are explicitly `class_weight="balanced"`.
- The final test split is not used for hyperparameter selection.

## Distributed Production Controls

```mermaid
sequenceDiagram
    participant Client
    participant API as API Replica
    participant Redis
    participant Worker
    participant DB

    Client->>API: POST /investigations/enqueue/{txn}
    API->>Redis: Atomic rate-limit check
    Redis-->>API: Allowed / 429
    API->>Redis: Create job + XADD stream
    API-->>Client: 202 + job_id + SLA deadline
    Worker->>Redis: XREADGROUP
    Redis-->>Worker: Investigation job
    Worker->>DB: Load transaction/evidence
    Worker->>Worker: ML + policy + agent investigation
    Worker->>DB: Persist report
    Worker->>Redis: Update job + XACK
    Client->>API: GET /investigations/jobs/{job_id}
    API->>Redis: Read job state
    Redis-->>API: Status/result
    API-->>Client: Job state/result
```

### Failure semantics

- API replicas share Redis rate-limit state.
- Investigation jobs are stored in a Redis Stream rather than an in-process queue.
- Consumer groups distribute jobs across worker replicas.
- Pending messages can be reclaimed after worker failure.
- Transient worker failures are retried, bounded by `INVESTIGATION_MAX_ATTEMPTS` and the SLA deadline.
- Job state has a TTL to prevent unbounded Redis growth.
- Synchronous investigation remains available for compatibility, but asynchronous investigation is the preferred production path.

---

## Current production data-plane override — PostgreSQL

The historical SQLite sections above document the bugs and migration history that led to the current architecture. The **current production implementation is PostgreSQL-backed**.

- `db/database.py` selects PostgreSQL whenever `DATABASE_URL` is a PostgreSQL URL.
- The raw connection helper name `get_raw_sqlite_connection()` is retained for backward-compatible application imports; it returns a PostgreSQL-compatible connection in production.
- `db/schema.sql` is PostgreSQL-first.
- `razor_risk.db` is not part of the production data path — it's a convenience artifact for the manual/no-Docker Quick Start path, which explicitly falls back to SQLite (see `README.md`'s Quick Start, option B).
- `docker-compose.yml` provides PostgreSQL locally; Supabase can provide the managed production instance.
- `tests/conftest.py` explicitly selects SQLite so the deterministic unit suite does not require external infrastructure.
- API replicas and investigation workers must share the same `DATABASE_URL`.

This makes Redis the distributed coordination plane and PostgreSQL the distributed durable application-data plane.
