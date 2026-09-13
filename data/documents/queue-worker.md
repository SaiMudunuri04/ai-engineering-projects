# Queue worker operations

If a queue worker is slow, inspect queue depth and processing latency first. Check whether
downstream dependencies are responding. Before a planned restart, drain in-flight work and
confirm the queue is empty. After restart, verify that throughput recovers and error rates fall.
