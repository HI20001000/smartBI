from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]


class SQLQueryExecutor:
    """Execute read-only SQL against MySQL using optional runtime dependency."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        connect_timeout: int = 5,
        read_timeout: int = 30,
    ):
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.database = database
        self.connect_timeout = int(connect_timeout)
        self.read_timeout = int(read_timeout)

    @staticmethod
    def _normalize_single_select_sql(sql: str) -> str | None:
        normalized = (sql or "").strip()
        if not normalized:
            return None

        # allow at most one trailing semicolon
        if normalized.endswith(";"):
            normalized = normalized[:-1].rstrip()

        # still contains semicolon means possible multi-statement
        if ";" in normalized:
            return None

        lowered = normalized.lower()
        if not lowered.startswith("select"):
            return None

        blocked = [" insert ", " update ", " delete ", " drop ", " alter ", " create "]
        wrapped = f" {lowered} "
        if any(token in wrapped for token in blocked):
            return None

        return normalized

    @staticmethod
    def _is_safe_select(sql: str) -> bool:
        return SQLQueryExecutor._normalize_single_select_sql(sql) is not None

    def run(self, sql: str, max_rows: int = 1000) -> QueryResult:
        normalized_sql = self._normalize_single_select_sql(sql)
        if not normalized_sql:
            raise ValueError("Only single SELECT queries are allowed.")

        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("pymysql is required for SQL execution. Please install dependency.") from exc

        limited_sql = normalized_sql
        if " limit " not in limited_sql.lower():
            limited_sql = f"{limited_sql}\nLIMIT {int(max_rows)}"

        conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            cursorclass=DictCursor,
            autocommit=True,
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(limited_sql)
                rows = cursor.fetchall() or []
                columns = list(rows[0].keys()) if rows else []
                return QueryResult(columns=columns, rows=list(rows))
        finally:
            conn.close()
