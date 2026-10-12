"""Prometheus-compatible metrics with no third-party dependency.

Exposed at ``GET /api/metrics`` in the text exposition format (version 0.0.4),
so any Prometheus, Grafana Agent, OpenTelemetry Collector or cloud managed
scraper (Amazon Managed Prometheus, Google Managed Prometheus, Azure Monitor)
can collect them.

Only counters, gauges and fixed-bucket histograms are needed; label sets are
bounded (route templates, not raw paths; tenant ids are capped by
``MAX_TENANTS``), so cardinality cannot grow without limit.
"""

from __future__ import annotations

import math
import threading
from typing import Callable, Iterable, Optional

LabelKey = tuple[tuple[str, str], ...]

DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


def _labels(labels: dict[str, str]) -> LabelKey:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def _fmt_labels(key: LabelKey, extra: Optional[tuple[str, str]] = None) -> str:
    items = list(key) + ([extra] if extra else [])
    if not items:
        return ""
    body = ",".join(f'{k}="{v.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34)).replace(chr(10), " ")}"'
                    for k, v in items)
    return "{" + body + "}"


def _fmt_value(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value)) if not float(value).is_integer() else str(int(value))


class Counter:
    def __init__(self, name: str, help_text: str) -> None:
        self.name, self.help = name, help_text
        self._values: dict[LabelKey, float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = _labels(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} counter"
        with self._lock:
            for key, value in sorted(self._values.items()):
                yield f"{self.name}{_fmt_labels(key)} {_fmt_value(value)}"


class Histogram:
    def __init__(self, name: str, help_text: str, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self.name, self.help, self.buckets = name, help_text, tuple(sorted(buckets))
        self._counts: dict[LabelKey, list[int]] = {}
        self._sums: dict[LabelKey, float] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = _labels(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[i] += 1
            counts[-1] += 1
            self._sums[key] = self._sums.get(key, 0.0) + value

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} histogram"
        with self._lock:
            for key in sorted(self._counts):
                counts = self._counts[key]
                for bound, count in zip(self.buckets, counts):
                    yield f"{self.name}_bucket{_fmt_labels(key, ('le', _fmt_value(bound)))} {count}"
                yield f"{self.name}_bucket{_fmt_labels(key, ('le', '+Inf'))} {counts[-1]}"
                yield f"{self.name}_sum{_fmt_labels(key)} {_fmt_value(self._sums[key])}"
                yield f"{self.name}_count{_fmt_labels(key)} {counts[-1]}"


class GaugeCallback:
    """A gauge whose samples are computed at scrape time."""

    def __init__(self, name: str, help_text: str, collect: Callable[[], Iterable[tuple[dict[str, str], float]]]) -> None:
        self.name, self.help, self.collect = name, help_text, collect

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} gauge"
        for labels, value in self.collect():
            yield f"{self.name}{_fmt_labels(_labels(labels))} {_fmt_value(value)}"


class Registry:
    def __init__(self) -> None:
        self._metrics: list = []

    def register(self, metric):  # type: ignore[no-untyped-def]
        self._metrics.append(metric)
        return metric

    def render(self) -> str:
        lines: list[str] = []
        for metric in self._metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"
