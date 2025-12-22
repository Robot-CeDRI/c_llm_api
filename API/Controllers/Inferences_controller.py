from fastapi import APIRouter
import time

from API.Utils.Setup_LLM import LLM_MODEL
from API.Utils.RAG_Engine import RAG_ENGINE
from API.Utils.Databases.SQL_Database import DATABASE
from API.Utils.GeneratedTextProcessing import process_generated_text

from API.DataTransferObjects.Requests.inferenceRequestDTO import InferenceRequestDTO
from API.DataTransferObjects.Responses.inferenceResponseDTO import InferenceResponseDTO
from API.DataTransferObjects.Common.messageDTO import MessageDTO

router = APIRouter()


def _norm(s: str) -> str:
    return " ".join((s or "").split())


@router.post(
    "/",
    summary="Execute the LLM Inference of a sequence of messages, identifying context with a RAG Engine.",
    response_model=InferenceResponseDTO
)
async def exec_inference(inference_data: InferenceRequestDTO):
    start = time.time()

    # Guardar pergunta original (para logging)
    original_user_question = inference_data.messages[-1].content if inference_data.messages else ""

    ctx = ""  # vamos usar para guardrail
    fallback = "I don't know based on the provided knowledge base."

    try:
        k = inference_data.rag_parameters.k
        if k and k > 0:
            rag_pack = await RAG_ENGINE.find_contexts(
                query=inference_data.messages[-1].content,
                k=k
            )

            print("[RAG] rag_pack is None/empty?", (not rag_pack))

            if rag_pack and "system_prompt" in rag_pack and "user_prompt" in rag_pack:
                inference_data.messages = [
                    MessageDTO(role="system", content=rag_pack["system_prompt"]),
                    MessageDTO(role="user", content=rag_pack["user_prompt"]),
                ]
            else:
                # Sem contexto -> segue normal (ou podes devolver logo "I don't know..." se quiseres)
                inference_data.messages = [
                    MessageDTO(role="system", content="You are an institutional assistant for IPB/CeDRI."),
                    MessageDTO(role="user", content=inference_data.messages[-1].content),
                ]

    except Exception as e:
        print(f"[RAG] disabled for this request due to error: {e}")

        ctx = ""

    inference = await LLM_MODEL.exec_inference(
        messages=inference_data.messages,
        response_num_tokens=inference_data.inference_parameters.tokens_count,
        response_temperature=inference_data.inference_parameters.temperature,
        top_k=inference_data.inference_parameters.top_k_tokens,
        top_p=inference_data.inference_parameters.top_p_tokens,
    )
    end = time.time()

    # Processar saída
    messages, gen_text = process_generated_text(inference.split("</s>"))

    # ✅ GUARDRAIL: se houve contexto, só aceitamos resposta que esteja dentro do CONTEXT
    if ctx and ctx.strip():
        if _norm(gen_text) != _norm(fallback):
            if _norm(gen_text) not in _norm(ctx):
                gen_text = fallback
                # opcional: também substitui a lista de mensagens para refletir
                messages = [
                    {"role": "system", "content": "You are an institutional assistant for IPB/CeDRI."},
                    {"role": "user", "content": original_user_question},
                    {"role": "assistant", "content": gen_text},
                ]

    # Guardar no DB
    DATABASE.add_operation(
        user_name=inference_data.user_name,
        system_response=gen_text,
        inference_data=inference_data
    )

    return InferenceResponseDTO(
        messages=messages,
        generated_text=gen_text,
        inference_time_in_seconds=end - start,
    )
