import asyncio
import warnings
import contextlib
import secrets
import time
import os

from dotenv import load_dotenv
load_dotenv()


from fastapi.concurrency import asynccontextmanager

from typing import Any, Dict
from fastapi import BackgroundTasks, FastAPI, WebSocket, Request, Depends, HTTPException
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware 
from fastapi_mail import FastMail, MessageSchema, MessageType

from apps.vision.basic_agent.agent import basic_agent
from apps.vision.pro_agent.agent import pro_agent
from apps.vision.config import EmailSchema, mail_config
from apps.vision.services.engine import AgentEngine  # NEW

from apps.vision.shared.auth import require_roles, TokenPayload

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


APP_NAME = "Vision AI"
FRONT_WEB = os.getenv("WEB_ORIGIN", "http://localhost:5173")       # Vite dev
FRONT_APP = os.getenv("APP_ORIGIN", "https://app.thevision.ai")    # Prod web
ALLOWED_ORIGINS = [FRONT_WEB, FRONT_APP]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.basic_agent = AgentEngine(app_name=APP_NAME, base_agent=basic_agent)
    app.state.pro_agent = AgentEngine(app_name=APP_NAME, base_agent=pro_agent)
    app.state.sessions = {}
    try:
        yield
    finally:
        # Shutdown
        if hasattr(app.state, "basic_agent"):
            with contextlib.suppress(Exception):
                await app.state.basic_agent.aclose()
        if hasattr(app.state, "pro_agent"):
            with contextlib.suppress(Exception):
                await app.state.pro_agent.aclose()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Configure properly for production
    allow_credentials=False,
    allow_methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"],
    allow_headers=["Authorization","Content-Type","Accept","X-Requested-With"],
    expose_headers=["Content-Length","Content-Type"], 
    max_age=86400,
)

SESSION_TTL = 60 * 10 # attach window (10 min). tune as you like


def get_agent_tier_from_roles(roles: list[str]) -> str:
    normalized_roles = {role.lower() for role in roles}
    if "admin" in normalized_roles:
        return "advanced"
    if "pro" in normalized_roles:
        return "pro"
    return "basic"


@app.post("/session/start")
async def start_session() -> JSONResponse:
    session_id = secrets.token_urlsafe(24)
    app.state.sessions[session_id] = {
        "created": time.time(),
        "user": None,
        "roles": [],
        "agent_session": None
    }
    return JSONResponse(status_code=200, content={"session_id": session_id})

@app.post("/session/attach")
async def attach_session(
    request: Request,
    user: TokenPayload = Depends(require_roles("basic, pro, admin"))
) -> JSONResponse:
    session_id = await request.json()
    session = app.state.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if (time.time() - session["created"]) > SESSION_TTL and session["user"] is None:
        raise HTTPException(status_code=401, detail="Session expired")
    
    claims = user.model_dump()
    tid = claims.get("tid")
    oid = claims.get("oid") or claims.get("sub")
    if not (tid and oid):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    roles = claims.get("roles") or []
    ext_key = os.getenv("EXT_ROLE_CLAIM")
    if ext_key and isinstance(claims.get(ext_key), str):
        roles = list(set(roles + [claims[ext_key]]))

    session.update({"user": (tid, oid), "roles": roles, "attached_at": time.time()})
    return JSONResponse(status_code=200, content={"ok": True})


@app.post("/stream/")
async def sse_endpoint(request: Request):
    """SSE endpoint for agent to communicate with client"""
    data = await request.json()
    session_id = data.get("session_id")
    initial_message = data.get("message")

    session = app.state.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tier = get_agent_tier_from_roles(session["roles"])
    agent = app.state.basic_agent if tier == "basic" else app.state.pro_agent

    # Create internal per-session engine session
    agent_session = await agent.create_session(
        user_id=(session["user"][1] if session["user"] else session_id)
    )
    
    # Store user session in state
    session["agent_session"] = agent_session
    print(f"SSE connected: tier={tier}")

    # send initial message if provided
    if initial_message:
        await agent_session.client_to_agent_sse(message=initial_message)
    
    async def stream():
        try:
            async for data in agent_session.agent_to_client_sse():
                yield data
        except Exception as e:
            print(f"SSE error sid={session_id}: {e}")
        finally:
            try:
                await agent_session.close()
            except Exception:
                pass
            session["agent_session"] = None
            print(f"Client #{session_id} disconnected from SSE")

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
    )

@app.post("/send")
async def send_message_sse(request: Request) -> JSONResponse:
    """Client -> agent messages"""

    data = await request.json()
    session_id = data.get("session_id")

    # Get live session for this user
    session = app.state.sessions.get(session_id)
    if not session or not session.get("agent_session"):
        raise HTTPException(status_code=404, detail="Session not found")
    

    await session["agent_session"].client_to_agent_sse(message=data)
    return JSONResponse(status_code=200, content={"message": "message sent!"})

@app.post("/feedback/send")
async def send_feedback(background_tasks: BackgroundTasks, request: Request) -> JSONResponse:
    data = await request.json()

    feedback_msg = data.get("feedback", "No feedback provided")

    html = f"<h1>User Feedback</h1>\n<p>{feedback_msg}</p>"

    message = MessageSchema(
        subject="Feedback about Vision AI",
        recipients=["themanhimseph@gmail.com"],
        body=html,
        subtype=MessageType.html
    )

    fm = FastMail(mail_config)

    background_tasks.add_task(fm.send_message, message)

    return JSONResponse(status_code=200, content={"message": "Feedback received!"})
