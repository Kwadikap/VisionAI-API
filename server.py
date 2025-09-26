
from dotenv import load_dotenv
load_dotenv()

import os, time, secrets, contextlib, warnings, jwt
from jwt import PyJWTError
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import asynccontextmanager
from apps.vision.services.engine import AgentEngine
from apps.vision.basic_agent.agent import basic_agent
from apps.vision.pro_agent.agent import pro_agent
from apps.vision.shared.auth import SESSION_TTL, TICKET_TTL_MIN, create_ticket, extract_roles_from_claims, extract_subject_ids, extract_ticket, get_agent_tier_from_roles, validate_access_token_raw, validate_ticket  

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

ENV = os.getenv("ENV") or "dev"
APP_NAME = "Vision AI"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.sessions = {}
    app.state.agents = {
        "basic":   AgentEngine(app_name=APP_NAME, base_agent=basic_agent),
        "pro":     AgentEngine(app_name=APP_NAME, base_agent=pro_agent),
    }
    try:
        yield
    finally:
        # Shutdown
            for agent in app.state.agents.values():
                with contextlib.suppress(Exception):
                    await agent.aclose()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization","Content-Type","Accept","X-Requested-With"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


@app.post("/session/init")
async def session_init(request: Request) -> JSONResponse:
    auth = request.headers.get("Authorization")
    body = {}
    if request.headers.get("content-type","").startswith("application/json"):
        with contextlib.suppress(Exception):
            body = await request.json()

    requested_sid = body.get("session_id") if isinstance(body, dict) else None
    reuse = False
    if requested_sid and requested_sid in app.state.sessions:
        session_id = requested_sid
        reuse = True
    else:
        session_id = secrets.token_urlsafe(24)
        app.state.sessions[session_id] = {
            "created": time.time(),
            "user": None,
            "roles": [],
            "tier": "basic",
            "ticket": None,
            "agent_session": None,
            "is_guest": True,
        }

    session = app.state.sessions[session_id]
    tier = session.get("tier","basic")
    is_guest = session.get("is_guest", True)
    sub = None
    roles = session.get("roles", [])

    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ",1)[1]
        try:
            claims = await validate_access_token_raw(token)
            roles = extract_roles_from_claims(claims.model_dump())
            tid, oid = extract_subject_ids(claims.model_dump())
            if tid and oid:
                sub = oid
                tier = get_agent_tier_from_roles(roles)
                is_guest = False
                session.update({
                    "user": (tid, oid),
                    "roles": roles,
                    "tier": tier,
                    "is_guest": False,
                    "attached_at": time.time()
                })
        except Exception as e:
            print(f"[session/init] auth invalid -> guest: {e}")

    ticket = create_ticket(sid=session_id, tier=tier, sub=sub)
    session["ticket"] = ticket

    print(f"Setting sse_ticket cookie with value: {ticket[:20]}...")

    resp = JSONResponse({
        "session_id": session_id,
        "tier": tier,
        "is_guest": is_guest,
        "roles": roles,
        "reused": reuse
    })
    resp.set_cookie(
        "sse_ticket",
        ticket,
        max_age=TICKET_TTL_MIN * 60,
        httponly=True,
        secure=False, # set to true in PROD
        samesite="Lax",
        path="/"
    )

    return resp

@app.get("/events/{session_id}")
async def sse_events(session_id: str, request: Request, ticket: Optional[str] = Query(None)):
    token = extract_ticket(request, ticket)
    claims = validate_ticket(token)
    if claims["sid"] != session_id:
        raise HTTPException(403, "Ticket session mismatch")

    session = app.state.sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if (time.time() - session["created"]) > SESSION_TTL and session["user"] is None:
        raise HTTPException(401, "Session expired")

    tier = claims["tier"]
    agent = app.state.agents.get(tier) or app.state.agents["basic"]

    if not session.get("agent_session"):
        agent_session = await agent.create_session(
            user_id=session["user"][1] if session["user"] else session_id
        )
        session["agent_session"] = agent_session

    async def stream():
        try:
            async for data in session["agent_session"].agent_to_client_sse():
                yield data
        except Exception as e:
            print(f"SSE error sid={session_id}: {e}")
        finally:
            with contextlib.suppress(Exception):
                await session["agent_session"].close()
            session["agent_session"] = None

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, request: Request, ticket: Optional[str] = Query(None)):
    token = extract_ticket(request, ticket)
    claims = validate_ticket(token)
    if claims["sid"] != session_id:
        raise HTTPException(403, "Ticket session mismatch")

    session = app.state.sessions.get(session_id)
    if not session or not session.get("agent_session"):
        raise HTTPException(404, "Session not found or not connected")

    msg = await request.json()
    await session["agent_session"].client_to_agent_sse(message=msg)
    return JSONResponse({"ok": True})
