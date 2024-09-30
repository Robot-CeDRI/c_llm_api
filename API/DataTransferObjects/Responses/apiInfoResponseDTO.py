from pydantic import BaseModel

class APIInfoDTO(BaseModel):
    version: str
    description: str
    base_model: str
    fine_tuned_model: str | None
    rag_model_path: str | None
    device: str