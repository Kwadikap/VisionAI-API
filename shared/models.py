from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatMessageDTO(BaseModel):
    id: str
    isUser: bool
    data: str
    type: str = "text/plain"

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessageDTO]