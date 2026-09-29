"""Thin wrapper around pyTigerGraph.

Centralizes connection + token handling and exposes the handful of operations
the rest of the app needs: run GSQL, run installed queries, upsert vertices /
edges / vector attributes.

Designed to fail loudly with a clear message when TigerGraph is unreachable,
so the rest of the system can be developed and unit-tested without a live
Savanna instance.
"""

from __future__ import annotations

from typing import Any

from app.config import settings


class TigerGraphError(RuntimeError):
    pass


class GraphClient:
    def __init__(self) -> None:
        self._conn = None

    # -- connection --------------------------------------------------------

    def connect(self) -> "GraphClient":
        try:
            import pyTigerGraph as tg
        except ImportError as e:  # pragma: no cover
            raise TigerGraphError(
                "pyTigerGraph is not installed. `pip install pyTigerGraph`."
            ) from e

        is_cloud = settings.tg_host.startswith("https://")

        if settings.tg_secret:
            # Savanna / cloud: authenticate with a database secret. No
            # username/password needed. pyTigerGraph exchanges the secret for
            # a REST++ token internally when gsqlSecret + tgCloud are set.
            conn = tg.TigerGraphConnection(
                host=settings.tg_host,
                graphname=settings.tg_graph,
                gsqlSecret=settings.tg_secret,
                tgCloud=is_cloud,
                restppPort=str(settings.tg_restpp_port),
                gsPort=str(settings.tg_gsql_port),
            )
            try:
                # Exchange the secret for a bearer token for REST++ calls.
                conn.getToken(settings.tg_secret)
            except Exception:  # noqa: BLE001 - some versions token via gsqlSecret alone
                pass
        else:
            # On-prem / local Docker: username + password.
            conn = tg.TigerGraphConnection(
                host=settings.tg_host,
                graphname=settings.tg_graph,
                username=settings.tg_username,
                password=settings.tg_password,
                tgCloud=is_cloud,
                restppPort=str(settings.tg_restpp_port),
                gsPort=str(settings.tg_gsql_port),
            )

        self._conn = conn
        return self

    @property
    def conn(self):
        if self._conn is None:
            self.connect()
        return self._conn

    def ping(self) -> bool:
        try:
            self.conn.echo()
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- schema / gsql -----------------------------------------------------

    @staticmethod
    def _is_auth_error(e: Exception) -> bool:
        msg = str(e).lower()
        return "authentic" in msg or "token" in msg or "expired" in msg

    def gsql(self, statement: str) -> str:
        try:
            return self.conn.gsql(statement)
        except Exception as e:  # noqa: BLE001
            if self._is_auth_error(e):
                # Stale/expired token -> reconnect (re-issues token) and retry.
                self.connect()
                try:
                    return self.conn.gsql(statement)
                except Exception as e2:  # noqa: BLE001
                    raise TigerGraphError(f"GSQL failed after re-auth: {e2}") from e2
            raise TigerGraphError(f"GSQL failed: {e}") from e

    def run_query(self, name: str, params: dict[str, Any] | None = None) -> list[dict]:
        """Run an installed query by name and return its raw result list."""
        try:
            return self.conn.runInstalledQuery(name, params or {}, timeout=120_000)
        except Exception as e:  # noqa: BLE001
            if self._is_auth_error(e):
                self.connect()
                try:
                    return self.conn.runInstalledQuery(name, params or {}, timeout=120_000)
                except Exception as e2:  # noqa: BLE001
                    raise TigerGraphError(f"runInstalledQuery({name}) failed after re-auth: {e2}") from e2
            raise TigerGraphError(f"runInstalledQuery({name}) failed: {e}") from e

    # -- upserts -----------------------------------------------------------

    def upsert_vertices(self, vertex_type: str, rows: list[tuple[str, dict]]) -> int:
        """rows = [(primary_id, {attr: value, ...}), ...]. Returns count upserted."""
        try:
            return self.conn.upsertVertices(vertex_type, rows)
        except Exception as e:  # noqa: BLE001
            raise TigerGraphError(f"upsertVertices({vertex_type}) failed: {e}") from e

    def upsert_edges(
        self,
        from_type: str,
        edge_type: str,
        to_type: str,
        rows: list[tuple[str, str, dict]],
    ) -> int:
        """rows = [(from_id, to_id, {attr: value}), ...]. Returns count upserted."""
        try:
            return self.conn.upsertEdges(from_type, edge_type, to_type, rows)
        except Exception as e:  # noqa: BLE001
            raise TigerGraphError(
                f"upsertEdges({from_type}-{edge_type}->{to_type}) failed: {e}"
            ) from e

    def upsert_document_embeddings(
        self, rows: list[tuple[str, list[float]]]
    ) -> int:
        """Upsert the `emb` vector attribute on Document vertices.

        rows = [(doc_id, [floats]), ...]
        Uses the RESTPP upsert payload so the vector attribute is set directly.
        """
        vertices: dict[str, dict] = {}
        for doc_id, vec in rows:
            vertices[doc_id] = {"emb": {"value": vec}}
        payload = {"vertices": {"Document": vertices}}
        try:
            return self.conn.upsertData(payload)
        except Exception as e:  # noqa: BLE001
            raise TigerGraphError(f"upsert embeddings failed: {e}") from e


_client: GraphClient | None = None


def get_client() -> GraphClient:
    global _client
    if _client is None:
        _client = GraphClient()
    return _client
