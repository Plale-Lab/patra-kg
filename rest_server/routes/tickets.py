import asyncpg
from fastapi import APIRouter, Depends, Path, Query, Request, status

from rest_server.database import get_pool
from rest_server.deps import get_request_actor, require_admin_actor
from rest_server.errors import not_found
from rest_server.workflow_models import TicketCreate, TicketRecord, TicketUpdate

router = APIRouter(tags=["tickets"])


def _row_to_ticket(row: asyncpg.Record) -> TicketRecord:
    return TicketRecord(
        id=str(row["id"]),
        subject=row["subject"],
        category=row["category"],
        priority=row["priority"],
        status=row["status"],
        description=row["description"],
        submitted_by=row["submitted_by"],
        submitted_at=row["submitted_at"],
        admin_response=row["admin_response"],
        updated_at=row["updated_at"],
        reviewed_by=row["reviewed_by"],
        reviewed_at=row["reviewed_at"],
    )


@router.get("/tickets", response_model=list[TicketRecord])
async def list_tickets(
    request: Request,
    pool: asyncpg.Pool = Depends(get_pool),
    status_filter: str | None = Query(default=None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    actor = get_request_actor(request)
    clauses: list[str] = []
    params: list[object] = []

    if status_filter:
        clauses.append(f"status = ${len(params) + 1}")
        params.append(status_filter)

    if actor.username and not actor.is_admin:
        clauses.append(f"submitted_by = ${len(params) + 1}")
        params.append(actor.username)
    elif not actor.is_admin and not actor.username:
        return []

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([limit, skip])
    query = f"""
        SELECT id, subject, category, priority, status, description,
               submitted_by, submitted_at, admin_response, updated_at,
               reviewed_by, reviewed_at
        FROM support_tickets
        {where}
        ORDER BY submitted_at DESC, id DESC
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [_row_to_ticket(row) for row in rows]


@router.post("/tickets", response_model=TicketRecord, status_code=status.HTTP_201_CREATED)
async def create_ticket(
    payload: TicketCreate,
    request: Request,
    pool: asyncpg.Pool = Depends(get_pool),
):
    actor = get_request_actor(request)
    submitted_by = actor.username or payload.submitted_by

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO support_tickets (
                subject, category, priority, status, description,
                submitted_by, submitted_at, updated_at
            )
            VALUES ($1, $2, $3, 'open', $4, $5, NOW(), NOW())
            RETURNING id, subject, category, priority, status, description,
                      submitted_by, submitted_at, admin_response, updated_at,
                      reviewed_by, reviewed_at
            """,
            payload.subject,
            payload.category,
            payload.priority,
            payload.description,
            submitted_by,
        )
    return _row_to_ticket(row)


@router.put("/tickets/{ticket_id}", response_model=TicketRecord)
async def update_ticket(
    payload: TicketUpdate,
    ticket_id: int = Path(..., ge=1),
    actor=Depends(require_admin_actor),
    pool: asyncpg.Pool = Depends(get_pool),
):
    reviewed_by = actor.username or "admin"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE support_tickets
            SET status = $2,
                admin_response = $3,
                reviewed_by = $4,
                reviewed_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            RETURNING id, subject, category, priority, status, description,
                      submitted_by, submitted_at, admin_response, updated_at,
                      reviewed_by, reviewed_at
            """,
            ticket_id,
            payload.status,
            payload.admin_response,
            reviewed_by,
        )
    if not row:
        raise not_found("Ticket")
    return _row_to_ticket(row)
