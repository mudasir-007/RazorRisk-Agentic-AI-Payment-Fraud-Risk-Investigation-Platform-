# RazorRisk Observability Runbook

## Endpoints

- API metrics: `GET /metrics`
- API health: `GET /health`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Worker metrics: `http://localhost:9101/metrics`

## Quick start

```bash
docker compose up --build
```

Then open Grafana and Prometheus. The Grafana Prometheus datasource is provisioned automatically.

## Useful PromQL

```promql
rate(razorrisk_http_requests_total[5m])
```

```promql
histogram_quantile(0.95, sum(rate(razorrisk_http_request_duration_seconds_bucket[5m])) by (le))
```

```promql
sum(razorrisk_investigations_total) by (status)
```

```promql
razorrisk_investigation_queue_depth
```

```promql
rate(razorrisk_rate_limit_hits_total[5m])
```

## Tracing

Set:

```env
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces
OTEL_SERVICE_NAME=razorrisk-api
```

For the worker, set `OTEL_SERVICE_NAME=razorrisk-worker`.

The FastAPI application is automatically instrumented when the OpenTelemetry packages are installed. The exporter is opt-in through `OTEL_EXPORTER_OTLP_ENDPOINT`.

## Jev verification signals

Jev exposes dedicated Prometheus counters, latency, retry, disagreement, grounding, unavailable, auto-resolution, and circuit-breaker metrics. Search the agent investigation
log for `[Jev]` entries to distinguish:

- verification consistency;
- recommended review due to action disagreement or weak hypothesis grounding;
- unavailable/missing API key;
- API/network/response failures.

For production rollout, benchmark Jev-enabled versus Jev-disabled investigation
latency with `tests/benchmarks/benchmark_jev.py` before treating the extra
verification call as part of an SLA budget.

## Failure interpretation

- **API errors rise + dependency failures rise:** inspect Redis/PostgreSQL/LLM dependencies.
- **Queue depth rises + investigation latency rises:** scale workers or investigate downstream latency.
- **Retries rise:** inspect worker crashes, Redis pending entries, database errors and LLM/provider timeouts.
- **Rate-limit hits rise:** inspect client identity, traffic bursts and configured limits.
- **Risk-tier distribution shifts:** investigate data distribution/model behavior before changing thresholds.

Observability should alert operators; it should not silently change fraud thresholds or payment decisions.
