from fastapi import APIRouter
import time
from datetime import datetime
from API.Utils.Setup_LLM import LLM_MODEL
from API.Utils.GeneratedTextProcessing import process_generated_text
from API.Utils.Cache_Database import CacheDatabase

from API.DataTransferObjects.Requests.inferenceRequestDTO import InferenceRequestDTO
from API.DataTransferObjects.Responses.inferenceResponseDTO import InferenceResponseDTO

router = APIRouter()

@router.post("/", summary="Execute the LLM Inference.", response_model=InferenceResponseDTO)
async def exec_inference(inference_data: InferenceRequestDTO):
    start = time.time()
    response = await LLM_MODEL.exec_inference(
        messages=inference_data.messages,
        response_num_tokens=inference_data.inference_parameters.tokens_count,
        response_temperature=inference_data.inference_parameters.temperature,
        top_k=inference_data.inference_parameters.top_k_tokens,
        top_p=inference_data.inference_parameters.top_p_tokens,
    )
    end = time.time()
    messages, gen_text = process_generated_text(response.split('</s>'))
    CacheDatabase().insert(inference_data, user_name=inference_data.user_name, last_message=gen_text, action="ExecInference")
    return InferenceResponseDTO(
        timestamp=datetime.now(),
        messages=messages,
        generated_text=gen_text,
        inference_time_in_seconds=end - start,
    )