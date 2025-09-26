# session_manager.py (hardened)

import json
import base64
from typing import Optional

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from google.adk.runners import InMemoryRunner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.genai import types
from google.genai.types import Part, Content, Blob, VoiceConfig, SpeechConfig, PrebuiltVoiceConfigDict

def _safe_send_text(ws: WebSocket, obj: dict):
    # Ensure we only ever push valid UTF-8 JSON in text frames
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    payload.encode("utf-8")  # will raise if invalid
    return ws.send_text(payload)

class AgentSessionManager:
    """Manages a single live agent session and its bidirectional messaging."""

    def __init__(
        self,
        runner: InMemoryRunner,
        run_config: RunConfig,
        app_name: str,
        user_id: str,
    ) -> None:
        self._runner = runner
        self._run_config = run_config
        self._app_name = app_name
        self._user_id = user_id

        self._live_request_queue: Optional[LiveRequestQueue] = None
        self._live_events = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        session = await self._runner.session_service.create_session(
            app_name=self._app_name,
            user_id=self._user_id,
        )

        self._live_request_queue = LiveRequestQueue()

        self._live_events = self._runner.run_live(
            session=session,
            live_request_queue=self._live_request_queue,
            run_config=self._run_config,
        )

        self._started = True

    async def agent_to_client_sse(self):
        """Agent to client communication via SSE"""
        assert self._live_events is not None, "Session not started"
        
        async for event in self._live_events:
            # Don't send tool calls and tool response back to the client
            if event.get_function_responses() or event.get_function_calls():
                continue
            # if event.is_final_response():
            #     if event.actions and event.actions.skip_summarization and event.get_function_responses():
            #         continue
            #     elif hasattr(event, 'long_running_tool_ids') and event.long_running_tool_ids:
            #         continue
            # send data if turn complete or interrupted
            if event.turn_complete or event.interrupted:
                message = {
                    "turn_complete": event.turn_complete,
                    "interrupted": event.interrupted,
                }
                yield f"data: {json.dumps(message)}\n\n"
                print(f"[AGENT TO CLIENT]: {message}")
                continue
            
            # Read content and first Part
            part: Part = (
                event.content and event.content.parts and event.content.parts[0]
            )
            if not part:
                continue

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
            
            # If text and partial text, send
            if part.text and event.partial:
                message = {
                    "mime_type": "text/plain",
                    "data": part.text
                }
                yield f"data: {json.dumps(message)}\n\n"
                print(f"[AGENT TO CLIENT]: text/plain: {message}")

    async def client_to_agent_sse(self, message):
        mime_type = message["mime_type"]
        data = message["data"]

        # Send message to the agent
        match (mime_type):
            case "text/plain":
                content = Content(role="user", parts=[Part.from_text(text=data)])
                self._live_request_queue.send_content(content=content)
                print(f"[CLIENT TO AGENT]: {data}")
            case "audio/pcm":
                decoded_data = base64.b64decode(data)
                self._live_request_queue.send_realtime(Blob(data=decoded_data, mime_type=mime_type))
                print(f"[CLIENT TO AGENT]: audio/pcm: {len(decoded_data)} bytes")
            case _:
                print(f"error: Mime type not supported: {mime_type}")
            


    async def agent_to_client(self, websocket: WebSocket) -> None:
        """Stream events from the agent to the client websocket."""
        assert self._live_events is not None, "Session not started"

        async for event in self._live_events:
            # Turn boundaries
            if event.turn_complete or event.interrupted:
                msg = {"turn_complete": event.turn_complete, "interrupted": event.interrupted}
                await _safe_send_text(websocket, msg)
                print(f"[AGENT TO CLIENT]: {msg}")
                continue

            part: Optional[Part] = (
                event.content and event.content.parts and event.content.parts[0]
            )
            if not part:
                continue

            # Inline data (bytes) from the model/tools (audio/images/pdf/others)
            inline = getattr(part, "inline_data", None)
            if inline and inline.mime_type and inline.data:
                # You already base64 audio; generalize to any inline bytes
                msg = {
                    "mime_type": inline.mime_type,
                    "data_b64": base64.b64encode(inline.data).decode("ascii"),
                }
                await _safe_send_text(websocket, msg)
                print(f"[AGENT TO CLIENT]: {inline.mime_type}: {len(inline.data)} bytes (b64).")
                continue

            # Plain text (partial or final)
            if part.text:
                if self._is_audio:
                    # In voice mode, suppress textual narration (keeps things simpler)
                    continue
                msg = {"mime_type": "text/plain", "data": part.text, "partial": bool(event.partial)}
                await _safe_send_text(websocket, msg)
                print(f"[AGENT TO CLIENT]: text/plain: {part.text[:120]!r}{'…' if len(part.text) > 120 else ''}")
                continue

            # (Anything else is ignored for now)

    async def client_to_agent(self, websocket: WebSocket) -> None:
        """Stream messages from the client websocket to the agent."""
        assert self._live_request_queue is not None, "Session not started"

        try:
            while True:
                msg = await websocket.receive()  # <-- crucial: handle text *and* binary
                if "text" in msg and msg["text"] is not None:
                    text = msg["text"]
                    # Validate UTF-8 + JSON to avoid 1007-inducing garbage
                    try:
                        text.encode("utf-8")
                        message = json.loads(text)
                    except (UnicodeError, json.JSONDecodeError):
                        print("[WS] dropped invalid text frame (utf8/json)")
                        continue

                    mime_type = message.get("mime_type")
                    data = message.get("data")

                    if mime_type == "text/plain":
                        content = Content(role="user", parts=[Part.from_text(text=data or "")])
                        self._live_request_queue.send_content(content=content)
                        print(f"[CLIENT TO AGENT]: text/plain: {str(data)[:120]!r}")
                        continue

                    if mime_type in ("audio/pcm", "audio/wav", "audio/opus"):
                        # Expect base64 if sent via text frames
                        b64 = data or message.get("data_b64")
                        if not b64:
                            print("[WS] dropped audio text frame without base64 payload")
                            continue
                        try:
                            decoded = base64.b64decode(b64)
                        except Exception:
                            print("[WS] dropped audio text frame (bad base64)")
                            continue
                        norm_mime = "audio/pcm;rate=16000" if mime_type.startswith("audio/pcm") else mime_type
                        self._live_request_queue.send_realtime(Blob(data=decoded, mime_type=norm_mime))
                        continue

                    else:
                        # Gracefully tolerate other JSON control messages if you add them later
                        if mime_type:
                            print(f"[CLIENT TO AGENT]: Unhandled mime_type in text frame: {mime_type}")
                        else:
                            print("[CLIENT TO AGENT]: Control JSON without mime_type; ignoring.")

                elif "bytes" in msg and msg["bytes"] is not None:
                    # If your UI ever sends raw binary (ArrayBuffer/Blob), accept it as realtime audio
                    # or route by an out-of-band header you define. This prevents 1007 on accidental binary.
                    raw = msg["bytes"]
                    if self._is_audio:
                        norm_mime = "audio/pcm;rate=16000"
                        self._live_request_queue.send_realtime(Blob(data=raw, mime_type=norm_mime))
                        print(f"[CLIENT TO AGENT]: binary audio {len(raw)} bytes")
                    else:
                        # If not in audio mode, reject or ignore binary explicitly (but don't 1007 the peer).
                        print(f"[CLIENT TO AGENT]: unexpected binary {len(raw)} bytes in TEXT mode; dropping.")
                        # Optionally: await websocket.close(code=1003, reason="Binary not allowed in TEXT mode")
                        # return

                elif msg.get("type") in {"websocket.disconnect", "websocket.close"}:
                    break

        except WebSocketDisconnect:
            print(f"[WS] client_to_agent disconnect: code={getattr(e, 'code', None)}")
            pass

    def close(self) -> None:
        if self._live_request_queue:
            self._live_request_queue.close()
            self._live_request_queue = None
        self._started = False
