from pydantic import BaseModel
from typing import List
from API.DataTransferObjects.Common.messageDTO import MessageDTO
from API.DataTransferObjects.Common.inferenceParametersDTO import InferenceParameters
from API.DataTransferObjects.Common.ragParametersDTO import RAGParameters

class InferenceRequestDTO(BaseModel):
    user_name: str | None
    messages: List[MessageDTO]
    inference_parameters: InferenceParameters
    rag_parameters: RAGParameters


