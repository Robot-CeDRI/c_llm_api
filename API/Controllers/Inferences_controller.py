from fastapi import APIRouter
import time
from datetime import datetime
from API.Utils.Setup_LLM import LLM_MODEL
from API.Utils.RAG_Engine import RAG_ENGINE
from API.Utils.Databases.SQL_Database import DATABASE
from API.Utils.GeneratedTextProcessing import process_generated_text


from API.DataTransferObjects.Requests.inferenceRequestDTO import InferenceRequestDTO
from API.DataTransferObjects.Responses.inferenceResponseDTO import InferenceResponseDTO

router = APIRouter()

@router.post("/", summary="Execute the LLM Inference of a sequence of messages, identifying context with a RAG Engine.", response_model=InferenceResponseDTO)
async def exec_inference(inference_data: InferenceRequestDTO):
    # 1. Execute the RAG_Engine context search in the documents from the knowledge database
    start = time.time()
    new_query = await RAG_ENGINE.find_contexts(query=inference_data.messages[-1].content, k=inference_data.rag_parameters.k)
    inference_data.messages[-1].content = new_query
    inference = await LLM_MODEL.exec_inference(
        messages=inference_data.messages,
        response_num_tokens=inference_data.inference_parameters.tokens_count,
        response_temperature=inference_data.inference_parameters.temperature,
        top_k=inference_data.inference_parameters.top_k_tokens,
        top_p=inference_data.inference_parameters.top_p_tokens,
    )
    end = time.time()

    # 2. Process the generated text and return in the messages format
    messages, gen_text = process_generated_text(inference.split('</s>'))

    # 3. Add operation to the database
    DATABASE.add_operation(user_name=inference_data.user_name, system_response=gen_text, inference_data=inference_data)

    # 4. Return the inference results
    return InferenceResponseDTO(
        messages=messages,
        generated_text=gen_text,
        inference_time_in_seconds=end - start,
    )