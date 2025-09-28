from apps.vision.services.session_manager import AgentSessionManager
from google.adk.runners import InMemoryRunner, Runner
from google.adk.artifacts import InMemoryArtifactService, GcsArtifactService
from google.adk.sessions import InMemorySessionService, VertexAiSessionService
from google.genai.types import Part, Content, Blob, VoiceConfig, SpeechConfig, PrebuiltVoiceConfigDict
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types
import os

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

class AgentEngine:
    """App-level engine to create live agent sessions (reuses one Runner)."""
    session_service = VertexAiSessionService(PROJECT_ID, LOCATION)

    def __init__(self, app_name: str, base_agent) -> None:

        # self.artifact_service = InMemoryArtifactService()

        self._runner = Runner(
            app_name=app_name, 
            agent=base_agent, 
            session_service=InMemorySessionService(), 
            # artifact_service=self.artifact_service
        )
        self._app_name = app_name

    async def create_session(self, user_id: str, is_audio: bool = False) -> AgentSessionManager:
        """Starts an agent session"""

        voice_config = VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfigDict(
                voice_name='Aoede'
            )
        )

        Speech_config = SpeechConfig(voice_config=voice_config)

        modality = "AUDIO" if is_audio else "TEXT"

        run_config = RunConfig(
            response_modalities=[modality],
            session_resumption=types.SessionResumptionConfig(),
            speech_config=Speech_config,
            streaming_mode=StreamingMode.SSE,
        )

        session = AgentSessionManager(
            runner=self._runner,
            run_config=run_config,
            app_name=self._app_name,
            user_id=user_id,
        )
        await session.start()
        return session

    async def aclose(self) -> None:
        # Placeholder for future resource cleanup if the runner needs it
        try:
            await self._runner.close()
        except Exception:
            pass
