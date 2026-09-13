# Queue worker latency

Inspect queue depth and p95 processing latency. If both rise, check downstream service errors.
Do not restart a worker before confirming in-flight work is drained. Escalate to the on-call
engineer when the queue continues to grow after the dependency recovers.
