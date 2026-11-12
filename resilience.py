"""Small, dependency-free resilience helpers used by graph and tool execution."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, Iterable, Optional


def classify_error(error: BaseException) -> set[str]:
    text = str(error).lower()
    kinds = {"transient_error"}
    if isinstance(error, (TimeoutError, TimeoutError)) or "timeout" in text:
        kinds.add("timeout")
    if "rate" in text and "limit" in text:
        kinds.add("rate_limit")
    if "schema" in text or "validation" in text or "json" in text:
        kinds.add("schema_violation")
    return kinds


def retry_call(
    operation: Callable[[], Any],
    config: Optional[Dict[str, Any]] = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Any:
    """Run an operation according to a YAML retry policy.

    ``max_attempts`` includes the initial call. A policy with no retryable
    categories still retries explicitly configured transient failures.
    """
    policy = config or {}
    attempts = max(1, int(policy.get("max_attempts", 1)))
    retry_on = set(policy.get("retry_on", ["timeout", "rate_limit", "transient_error", "schema_violation"]))
    initial = max(0.0, float(policy.get("initial_delay_s", 1)))
    maximum = max(0.0, float(policy.get("max_delay_s", 30)))
    jitter = float(policy.get("jitter", 0))

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt >= attempts or not (classify_error(error) & retry_on):
                raise
            delay = min(maximum, initial * (2 ** (attempt - 1)))
            if jitter:
                delay += delay * jitter * random_value()
            if on_retry:
                on_retry(attempt, error, delay)
            if delay:
                sleep(delay)
    raise RuntimeError("retry operation exited unexpectedly")


class TokenBucketRateLimiter:
    """Process-local token bucket limiter for configured tool calls."""

    def __init__(self, calls: int, per_seconds: float, clock: Callable[[], float] = time.monotonic):
        import threading

        self.capacity = max(1, int(calls))
        self.per_seconds = max(0.001, float(per_seconds))
        self.clock = clock
        self._tokens = float(self.capacity)
        self._last_refill = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self.clock()
                refill_rate = self.capacity / self.per_seconds
                self._tokens = min(self.capacity, self._tokens + (now - self._last_refill) * refill_rate)
                self._last_refill = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait_for = (1 - self._tokens) / refill_rate
            time.sleep(max(0.0, wait_for))


SlidingWindowRateLimiter = TokenBucketRateLimiter
