from __future__ import annotations

import re


def has_metric_name_column_reference(sql: str, selected_metrics: list[str]) -> bool:
    """Detect likely-invalid references like alias.metric_name in generated SQL.

    Metric names in semantic plans are logical names, not necessarily physical columns.
    If LLM emits `<table_alias>.<metric_name>` it often leads to MySQL 1054 unknown-column.
    """

    text = (sql or "").lower()
    if not text:
        return False

    metric_names = []
    for canonical in selected_metrics or []:
        if not isinstance(canonical, str) or "." not in canonical:
            continue
        metric_names.append(canonical.split(".", 1)[1].strip().lower())

    for metric_name in metric_names:
        if not metric_name:
            continue
        # matches `alias.metric_name` while avoiding false positives in long identifiers
        pattern = rf"\b[a-zA-Z_][\w]*\s*\.\s*{re.escape(metric_name)}\b"
        if re.search(pattern, text):
            return True

    return False

