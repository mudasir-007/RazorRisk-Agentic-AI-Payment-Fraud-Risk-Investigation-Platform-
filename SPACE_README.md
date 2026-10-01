---
- Optional Jev verification runs after investigation, outside the fraud-scoring hot path. It independently checks the investigator action and hypothesis grounding and can auto-resolve eligible non-mandatory HITL cases only when the configured confidence floor is met.
title: RazorRisk — Agentic Fraud Investigation Platform
emoji: 🕸️
colorFrom: red
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# RazorRisk

Agentic AI payment fraud & risk investigation platform — GraphSAGE GNN + tabular XGBoost + LLM investigation agent, combined by a learned stacker.

Open `/dashboard/` for the interactive demo, or `/docs` for the API.

Full source, architecture, and engineering write-up: https://github.com/YOUR_USERNAME/YOUR_REPO


## RazorRisk runtime notes

- The application uses a single raw SQLite data layer. `db/models.py`/an application-owned SQLAlchemy ORM path is not part of the runtime.
- Hourly velocity has two explicit dashboard modes: **ON** trusts the client value for controlled simulation; **OFF** calculates the trailing one-hour count in the backend. Production integrations should prefer backend mode.
- `HUMAN_REVIEW` creates a real `human_reviews` queue item after the transaction/risk record is committed. Reviewers resolve it through the HITL API/dashboard.
- The live GNN snapshot is invalidated after each committed transaction so rapid follow-up transactions see current topology without allowing a transaction to influence its own GNN score.
- See `README.md`, `PROJECT_WORKFLOW.md`, and `tests/GOLDEN_TEST_MATRIX.md` for the authoritative workflow, known bugs, and disclosed test gaps.
