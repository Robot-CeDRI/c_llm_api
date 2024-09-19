from pydantic import BaseModel

class APIInfoDTO(BaseModel):
    version: str
    description: str
    base_model: str
    fine_tuned_model: str | None
    device: str