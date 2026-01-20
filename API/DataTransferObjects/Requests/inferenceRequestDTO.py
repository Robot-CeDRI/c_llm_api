from pydantic import BaseModel
from typing import List, Optional
from API.DataTransferObjects.Common.messageDTO import MessageDTO
from API.DataTransferObjects.Common.inferenceParametersDTO import InferenceParameters
from API.DataTransferObjects.Common.ragParametersDTO import RAGParameters

class InferenceRequestDTO(BaseModel):
    user_name: Optional[str] = None
    messages: List[MessageDTO]
    inference_parameters: InferenceParameters = InferenceParameters()
    rag_parameters: RAGParameters = RAGParameters()
