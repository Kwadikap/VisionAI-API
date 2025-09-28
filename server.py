
from dotenv import load_dotenv
from fastapi_mail import FastMail, MessageSchema, MessageType

from apps.vision.services.session_service import SessionService
from apps.vision.shared.types import Tier
load_dotenv()

import os, time, secrets, contextlib, warnings
from fastapi import BackgroundTasks, FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import asynccontextmanager
from apps.vision.config import mail_config
from apps.vision.shared.auth import SESSION_TTL, TICKET_TTL_MIN, create_token, extract_roles_from_claims, extract_subject_ids, extract_ticket, extract_token, get_agent_tier_from_roles, validate_access_token_raw, validate_ticket, validate_token  

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

ENV = os.getenv("ENV") or "dev"
GUEST_USER = os.getenv("GUEST_USER")
RECIPIENT = os.getenv("RECIPIENT")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.sessions = {}
    app.state.hosts = {}
    try:
        yield
    finally:
       pass

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
    token = extract_token(request)
    # If token exists don't re-initialize
    if token: return 

    auth = request.headers.get("Authorization")

    tier = Tier.BASIC.value
    user_id = GUEST_USER + secrets.token_urlsafe(24)

    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ",1)[1]
        try:
            claims = await validate_access_token_raw(token)
            roles = extract_roles_from_claims(claims.model_dump())
            tid, oid = extract_subject_ids(claims.model_dump())
            if tid and oid:
                tier = get_agent_tier_from_roles(roles)
                user_id = str((tid, oid))

        except Exception as e:
            raise HTTPException(401,"[session/init] auth invalid -> guest: {e}")

    # Create session
    session = await SessionService.create_session(user_id=user_id)
    # Create token
    token = create_token(sid=session.id, tier=tier, sub=None)
    # Store detials in app state
    app.state.sessions[session.id] = {
        "user_id": user_id,
        "tier": tier,
        "session_id": session.id,
        "created_at": time.time(),
    }

    resp = JSONResponse(status_code=200, content={"message": "Session created"})

    resp.set_cookie(
        "token",
        token,
        max_age=None if auth else TICKET_TTL_MIN * 60,
        httponly=True,
        secure=True if ENV == "PROD" else False, 
        samesite="Lax",
        path="/"
    )

    return resp

@app.get("/stream")
async def sse_events(request: Request):
    token = extract_token(request)
    if not token:
        raise HTTPException(401, "Missing token")
    
    claims = validate_token(token)

    session_id = claims["sid"]

    session_data = app.state.sessions.get(session_id)
    if not session_data:
        raise HTTPException(404, "Session not found")
    if (time.time() - session_data["created_at"]) > SESSION_TTL and session_data["user_id"].startswith(GUEST_USER):
        raise HTTPException(401, "Session expired")

    user_id = session_data["user_id"]
    tier = claims["tier"]

    # Get session object
    session = await SessionService.get_session(user_id, session_id)
    # Start the session
    live_events, live_request_queue = await SessionService.start_session(session=session, tier=tier, voice_chat=False)

    # store request queue in state
    session_data["live_request_queue"] = live_request_queue

    async def stream():
        try:
            async for data in SessionService.stream_sse_agent_to_client(live_events):
                yield data
        except Exception as e:
            print(f"SSE error: {e}")
        finally:
            with contextlib.suppress(Exception):
                await live_request_queue.close()

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/send")
async def send_message(request: Request):
    token = extract_token(request)
    if not token:
        raise HTTPException(401, "Missing token")
    
    claims = validate_token(token)
    session_id = claims["sid"]

    session_data = app.state.sessions.get(session_id)
    if not session_data or not session_data.get("live_request_queue"):
        raise HTTPException(404, "Session not found or not connected")

    live_request_queue = session_data.get("live_request_queue")
    request = await request.json()
    await SessionService.client_to_agent_sse(live_request_queue, request)
    return JSONResponse({"Message sent": True})

@app.post("/feedback/send")
async def send_feedback(background_tasks: BackgroundTasks, request: Request) -> JSONResponse:
    data = await request.json()

    feedback_msg = data.get("feedback", "No feedback provided")

    html = f"<h1>User Feedback</h1>\n<p>{feedback_msg}</p>"

    message = MessageSchema(
        subject="Feedback about Vision AI",
        recipients=[RECIPIENT],
        body=html,
        subtype=MessageType.html
    )

    fm = FastMail(mail_config)

    background_tasks.add_task(fm.send_message, message)

    return JSONResponse(status_code=200, content={"message": "Feedback received!"})

