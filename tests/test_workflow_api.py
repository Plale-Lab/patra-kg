import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest

from rest_server.database import get_pool
from rest_server.deps import get_admin_users
from rest_server.main import app


class MockWorkflowConn:
    def __init__(self):
        self.ticket_id_seq = 0
        self.tickets: list[dict] = []

    async def fetch(self, query: str, *args):
        if "FROM support_tickets" in query:
            return self._filter_rows(self.tickets, query, *args)
        return []

    async def fetchrow(self, query: str, *args):
        if "INSERT INTO support_tickets" in query:
            self.ticket_id_seq += 1
            now = datetime.now(timezone.utc)
            row = {
                "id": self.ticket_id_seq,
                "subject": args[0],
                "category": args[1],
                "priority": args[2],
                "status": "open",
                "description": args[3],
                "submitted_by": args[4],
                "submitted_at": now,
                "admin_response": None,
                "updated_at": now,
                "reviewed_by": None,
                "reviewed_at": None,
            }
            self.tickets.append(row)
            return row
        if "UPDATE support_tickets" in query:
            ticket = self._find_by_id(self.tickets, args[0])
            if not ticket:
                return None
            ticket.update({
                "status": args[1],
                "admin_response": args[2],
                "reviewed_by": args[3],
                "reviewed_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })
            return ticket
        return None

    async def fetchval(self, query: str, *args):
        return None

    async def execute(self, query: str, *args):
        return None

    async def executemany(self, query: str, rows):
        return None

    @asynccontextmanager
    async def transaction(self):
        yield self

    @staticmethod
    def _find_by_id(items: list[dict], item_id: int):
        for item in items:
            if int(item["id"]) == int(item_id):
                return item
        return None

    @staticmethod
    def _filter_rows(items: list[dict], query: str, *args):
        params = list(args)
        limit = params[-2]
        offset = params[-1]
        filters = params[:-2]
        rows = list(items)

        if "status =" in query and filters:
            rows = [item for item in rows if item["status"] == filters[0]]
            filters = filters[1:]

        if "submitted_by =" in query and filters:
            rows = [item for item in rows if item["submitted_by"] == filters[0]]

        rows.sort(key=lambda item: (item["submitted_at"], item["id"]), reverse=True)
        return rows[offset : offset + limit]


class MockWorkflowPool:
    def __init__(self, conn: MockWorkflowConn):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


@pytest.fixture()
def workflow_client(monkeypatch):
    conn = MockWorkflowConn()
    pool = MockWorkflowPool(conn)
    monkeypatch.delenv("PATRA_ADMIN_USERS", raising=False)
    get_admin_users.cache_clear()

    @asynccontextmanager
    async def _no_op_lifespan(_):
        yield

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _no_op_lifespan
    app.dependency_overrides[get_pool] = lambda: pool

    with TestClient(app) as client:
        yield client, conn

    app.dependency_overrides.clear()
    app.router.lifespan_context = original_lifespan
    get_admin_users.cache_clear()


def test_ticket_create_and_admin_update(workflow_client):
    client, conn = workflow_client

    create_response = client.post(
        "/tickets",
        json={
            "submitted_by": "alice",
            "subject": "Cannot access private models",
            "category": "Access Request",
            "priority": "High",
            "description": "I need access to the private model cards for my team.",
        },
    )
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["subject"] == "Cannot access private models"
    assert created["status"] == "open"
    assert created["submitted_by"] == "alice"

    list_response = client.get("/tickets?limit=10&offset=0")
    assert list_response.status_code == 200
    tickets = list_response.json()
    assert len(tickets) == 1
    assert tickets[0]["id"] == str(created["id"])

    update_response = client.put(
        f"/tickets/{created['id']}",
        json={"status": "resolved", "admin_response": "Access granted."},
        headers={"X-Tapis-Token": "tok", "X-Patra-Username": "williamq96"},
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["status"] == "resolved"
    assert updated["reviewed_by"] == "williamq96"
    assert updated["admin_response"] == "Access granted."
