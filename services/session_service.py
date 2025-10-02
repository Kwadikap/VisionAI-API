from typing import Optional
from dotenv import load_dotenv

from apps.vision.shared.models import ChatHistoryResponse, ChatMessageDTO
load_dotenv()
from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService, InMemorySessionService
from google.genai.types import Part, Content, Blob, VoiceConfig, SpeechConfig, PrebuiltVoiceConfigDict
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.agents import LiveRequestQueue
from apps.vision.basic_agent.agent import basic_agent
from apps.vision.pro_agent.agent import pro_agent
from google.genai import types
from google.adk.events import Event
from google.genai.types import Content, Part
import time
import os
import json
import base64

from apps.vision.shared.types import Tier

APP_NAME = os.getenv("APP_NAME")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

class SessionService:
    service = InMemorySessionService()

    @staticmethod
    async def create_session(user_id):
        session = await SessionService.service.create_session(
            app_name=APP_NAME, 
            user_id=user_id)
        return session
    
    @staticmethod
    async def list_sessions(user_id):
        sessions = await SessionService.service.list_sessions(
            app_name=APP_NAME, 
            user_id=user_id)
        return sessions
    
    @staticmethod
    async def get_session(user_id, session_id):
        session = await SessionService.service.get_session(
            app_name=APP_NAME, 
            user_id=user_id, 
            session_id=session_id)
        return session
    
    @staticmethod
    async def delete_session(user_id, session_id):
        try:
            await SessionService.service.delete_session(
                app_name=APP_NAME, 
                user_id=user_id, 
                session_id=session_id)
        except Exception:
            print("Error occured during session deletion")
            pass

    @staticmethod
    async def start_session(session, tier, voice_chat):
        agent = pro_agent if tier == Tier.PRO.value else basic_agent

        # Setup run config
        speech_config = None
        modality = "TEXT"
        if voice_chat:
            voice_config = SessionService.get_voice_config()
            speech_config = SpeechConfig(voice_config=voice_config)
            modality = "AUDIO"

        run_config = RunConfig(
            response_modalities=[modality],
            session_resumption=types.SessionResumptionConfig(),
            speech_config=speech_config,
            streaming_mode=StreamingMode.SSE
        )

        # Create a Runner
        runner = Runner(
            app_name=APP_NAME,
            agent=agent,
            session_service=SessionService.service
        )

        # create live request queue
        live_request_queue = LiveRequestQueue()

        # Start session
        live_events = runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config
        )

        return live_events, live_request_queue
    
    @staticmethod
    async def stream_sse_agent_to_client(live_events):
        assert live_events is not None, "Session not started"

        async for event in live_events:
            # Don't send tool calls and tool response back to client
            if event.get_function_responses() or event.get_function_calls():
                continue
            
            # Send data if turn complete or interrupted:
            if event.turn_complete or event.interrupted:
                message = {
                    "turn_complete": event.turn_complete,
                    "interrupted": event.interrupted,
                }
                yield f"data: {json.dumps(message)}\n\n"
                print(f"[AGENT TO CLIENT]: {message}")
                continue
                
            # Read content and first part
            part: Part = (
                event.content and event.content.parts and event.content.parts[0]
            )
            if not part: continue

            # If text and partial text, send
            if part.text and event.partial:
                message = {
                    "mime_type": "text/plain",
                    "data": part.text
                }
                yield f"data: {json.dumps(message)}\n\n"
                print(f"[AGENT TO CLIENT]: text/plain: {message}")

            # If audio, send Base64 encoded audio data
            is_audio = part.inline_data and part.inline_data.mime_type.startswith("audio/pcm")
            if is_audio:
                audio_data = part.inline_data and part.inline_data.data
                if audio_data:
                    message = {
                        "mime_type": "audio/pcm",
                        "data": base64.b64encode(audio_data).decode("ascii")
                    }
                    yield f"data: {json.dumps(message)}\n\n"
                    print(f"[AGENT TO CLIENT]: audio/pcm: {len(audio_data)} bytes.")
                    continue

    @staticmethod
    async def client_to_agent_sse(live_request_queue, request):
        mime_type = request.get("mime_type")
        data = request.get("data")

        # Send message to the agent
        match (mime_type):
            case "text/plain":
                content = Content(role="user", parts=[Part.from_text(text=data)])
                live_request_queue.send_content(content=content)
                print(f"[CLIENT TO AGENT]: {data}")
            case "audio/pcm":
                decoded_data = base64.b64decode(data)
                live_request_queue.send_realtime(Blob(data=decoded_data, mime_type=mime_type))
                print(f"[CLIENT TO AGENT]: audio/pcm: {len(decoded_data)} bytes")
            case _:
                print(f"Mime type not supported: {mime_type}")


    @staticmethod
    def get_voice_config():
        return VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfigDict(
                voice_name='Aoede'
            )
        )
    
    @staticmethod
    async def parse_session_data(sessions):
        """
        sessions: output of InMemorySessionService.list_sessions(app_name, user_id)
        Shape: iterable of (user_id, [session_objects])
        Returns: list[ChatHistoryResponse], ordered newest-first
        """
        results_with_ts: list[tuple[float, ChatHistoryResponse]] = []

        # Helpers ---------------------------------------------------------------
        def _parts_text(content) -> str:
            if not content or not getattr(content, "parts", None):
                return ""
            out = []
            for p in content.parts:
                t = getattr(p, "text", None)
                if t:
                    out.append(t)
            return "".join(out)

        def _to_is_user(role: Optional[str]) -> bool:
            # ADK uses 'model' for assistant; 'user' for the user
            return (role or "").lower() == "user"

        # ----------------------------------------------------------------------
        for _, session_list in sessions:
            for data in session_list:
                # Always fetch the full session so we have its events
                session = await SessionService.get_session(session_id=data.id, user_id=data.user_id)
                events = getattr(session, "events", []) or []

                messages: list[ChatMessageDTO] = []

                # Streaming coalescer buffer
                buffer_text: list[str] = []
                buffer_role: Optional[str] = None
                buffer_event_id: Optional[str] = None
                buffer_created: Optional[float] = None
                buffer_invocation: Optional[str] = None  # optional boundary by invocation

                def flush():
                    nonlocal buffer_text, buffer_role, buffer_event_id, buffer_created, buffer_invocation
                    if not buffer_text:
                        return
                    text = "".join(buffer_text).strip()
                    if not text:
                        buffer_text.clear()
                        buffer_role = buffer_event_id = buffer_created = buffer_invocation = None
                        return
                    messages.append(
                        ChatMessageDTO(
                            id=buffer_event_id or f"msg_{len(messages)}",
                            isUser=_to_is_user(buffer_role),
                            data=text,
                            type="text/plain",
                            created_at=buffer_created,
                        )
                    )
                    buffer_text.clear()
                    buffer_role = buffer_event_id = buffer_created = buffer_invocation = None

                # Iterate in chronological order (defensive)
                for ev in sorted(events, key=lambda e: getattr(e, "timestamp", 0) or 0):
                    content = getattr(ev, "content", None)
                    role = getattr(content, "role", None)  # 'user' | 'model' | maybe 'tool'
                    text = _parts_text(content)
                    turn_done = bool(getattr(ev, "turn_complete", False) or getattr(ev, "interrupted", False))
                    inv = getattr(ev, "invocation_id", None)

                    # Boundary: if invocation changes (optional), close previous buffer
                    if buffer_invocation is not None and inv and inv != buffer_invocation:
                        flush()

                    # If this event carries text, treat it as a chunk (delta or full)
                    if text:
                        # Boundary: role switch closes previous buffered message
                        if buffer_role is not None and role != buffer_role:
                            flush()
                        if buffer_role is None:
                            buffer_role = role
                            buffer_event_id = getattr(ev, "id", None)
                            buffer_created = getattr(ev, "timestamp", None)
                            buffer_invocation = inv
                        buffer_text.append(text)

                    # If the turn is explicitly complete (or interrupted), flush
                    if turn_done and buffer_text:
                        flush()

                # Final flush if any residue
                flush()

                # Build response object
                resp = ChatHistoryResponse(
                    session_id=session.id,
                    messages=messages,
                )

                # Determine last-activity timestamp for this session
                last_ts = (
                    getattr(data, "last_update_time", None)
                    or getattr(session, "last_update_time", None)
                    or (messages[-1].created_at if messages else None)
                    or (max((getattr(e, "timestamp", 0) or 0) for e in events) if events else 0.0)
                )
                # Normalize to float
                last_ts = float(last_ts or 0.0)

                results_with_ts.append((last_ts, resp))

        # Sort newest-first and return only the payloads
        results_with_ts.sort(key=lambda t: t[0], reverse=True)
        return [resp for _, resp in results_with_ts]
    
    @staticmethod
    async def append_event(*, user_id: str, session_id: str, event: Event):
        """
        Backend-agnostic append that works with InMemorySessionService and others.
        Tries common signatures, then falls back to in-memory mutation.
        """
        service = SessionService.service
        # Try the canonical service signature first
        if hasattr(service, "append_event"):
            try:
                return await service.append_event(
                    app_name=APP_NAME,
                    user_id=user_id,
                    session_id=session_id,
                    event=event,
                )
            except TypeError:
                # Some impls accept (session, event)
                try:
                    session = await service.get_session(
                        app_name=APP_NAME, user_id=user_id, session_id=session_id
                    )
                    return await service.append_event(session=session, event=event)
                except Exception:
                    pass

        # Fallback: mutate in place (works for InMemorySessionService)
        session = await service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        if hasattr(session, "events") and isinstance(session.events, list):
            session.events.append(event)
            if hasattr(service, "update_session"):
                try:
                    await service.update_session(session=session)
                except Exception:
                    pass
            return event

        raise RuntimeError("No compatible append method found on session service.")

    
    @staticmethod
    async def append_user_message(
        *, 
        user_id: str,
        session_id: str,
        text: str,
        role: str = "user",       
        author: str | None = "user",
        turn_complete: bool = True,
        timestamp: float | None = None,
    ):
        
        """
        High-level helper: create an ADK Event from simple args and append it.
        """
        if not text or not text.strip():
            return None

        event = Event(
            content=Content(parts=[Part.from_text(text=text)], role=role),
            author=author or (user_id if role == "user" else role),
            turn_complete=turn_complete,
            timestamp=timestamp or time.time(),
        )
        return await SessionService.append_event(
            user_id=user_id, session_id=session_id, event=event
        )
