# Incident triage agent · bounded tool use

A plan–act–observe agent for incident triage. The model can search local runbooks, read an allowlisted metric, or finish with a summary. The executor rejects unrecognized actions, caps runbook output, limits the step budget, and records every decision and observation. It has **no mutation tool**, so it cannot restart or delete infrastructure.

Run against a local OpenAI-compatible chat server:

```bash
python3 -m portfolio_ai.agents "Queue latency is rising" \
  --runbooks data/runbooks --metrics data/metrics_synthetic.json \
  --model YOUR_LOCAL_MODEL --base-url http://127.0.0.1:8000/v1
```

The included metric values are synthetic. Model-provided summaries still need human review; the audit trace supports that review. For live operations, add authenticated telemetry readers, redaction, and operator approval before any write-capable tool is introduced.
