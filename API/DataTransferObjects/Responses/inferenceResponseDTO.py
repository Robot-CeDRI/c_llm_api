from pydantic import BaseModel
from typing import List
from datetime import datetime
from API.DataTransferObjects.Common.messageDTO import MessageDTO

class InferenceResponseDTO(BaseModel):
    timestamp: datetime
    generated_text: str
    messages: List[MessageDTO]
    inference_time_in_seconds: float