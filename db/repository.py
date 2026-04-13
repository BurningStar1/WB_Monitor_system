from __future__ import annotations

from typing import Any
from sqlalchemy import text
from sqlalchemy.engine import Engine


class RawRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def insert_raw_payload(
        self,
        endpoint: str,
        request_params: dict[str, Any],
        payload: list[dict[str, Any]] | dict[str, Any],
        source: str,
    ) -> None:
        stmt = text(
            """
            INSERT INTO raw.wb_api_payloads (
                source, endpoint, request_params, raw_payload
            ) VALUES (
                :source, :endpoint, CAST(:request_params AS jsonb), CAST(:raw_payload AS jsonb)
            )
            """
        )
        import json

        with self.engine.begin() as conn:
            conn.execute(
                stmt,
                {
                    "source": source,
                    "endpoint": endpoint,
                    "request_params": json.dumps(request_params, ensure_ascii=False),
                    "raw_payload": json.dumps(payload, ensure_ascii=False),
                },
            )
