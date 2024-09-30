from pydantic import BaseModel

class RAGParameters(BaseModel):
    k: int
