from pydantic import BaseModel, conint

class RAGParameters(BaseModel):
    k: conint(ge=0, le=10) = 0
