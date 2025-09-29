from dotenv import load_dotenv
load_dotenv()
from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService, InMemorySessionService
from google.genai.types import Part, Content, Blob, VoiceConfig, SpeechConfig, PrebuiltVoiceConfigDict
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.agents import LiveRequestQueue
from apps.vision.basic_agent.agent import basic_agent
from apps.vision.pro_agent.agent import pro_agent
from google.genai import types
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
    async def list_session_ids(user_id):
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
    