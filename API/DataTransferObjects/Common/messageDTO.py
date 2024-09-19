from pydantic import BaseModel
from typing import Literal

class MessageDTO(BaseModel):
    role: Literal['system', 'user']
    content: str