from fastapi import APIRouter
import time
import re

from API.Utils.Setup_LLM import LLM_MODEL
from API.Utils.Setup_LLM_LORA import LORA_MODEL
from API.Utils.RAG_Engine import RAG_ENGINE
from API.Utils.Databases.SQL_Database import DATABASE
from API.Utils.GeneratedTextProcessing import process_generated_text

from API.DataTransferObjects.Requests.inferenceRequestDTO import InferenceRequestDTO
from API.DataTransferObjects.Responses.inferenceResponseDTO import InferenceResponseDTO
from API.DataTransferObjects.Common.messageDTO import MessageDTO

router = APIRouter()

FALLBACK = "I don't know based on the provided knowledge base."
SYSTEM_DEFAULT = "You are an institutional assistant for IPB/CeDRI."

def _is_unknown(text: str) -> bool:
    t = (text or "").lower()
    return ("i don't know" in t) or ("i do not know" in t) or ("not enough information" in t)

def _make_public_messages(original_user_question: str, gen_text: str):
    return [
        {"role": "system", "content": SYSTEM_DEFAULT},
        {"role": "user", "content": original_user_question},
        {"role": "assistant", "content": gen_text},
    ]

@router.post("/", summary="Execute the LLM Inference...", response_model=InferenceResponseDTO)
async def exec_inference(inference_data: InferenceRequestDTO):
    start = time.time()

    original_user_question = inference_data.messages[-1].content if inference_data.messages else ""
    ctx = ""
    used_rag = False
    model_to_use = LLM_MODEL  # default: modelo geral (sem LoRA)

    # -------------------------
    # 1) RAG (se k > 0)
    # -------------------------
    try:
        k = inference_data.rag_parameters.k if inference_data.rag_parameters else 0

        if k and k > 0 and inference_data.messages:
            rag_pack = await RAG_ENGINE.find_contexts(
                query=inference_data.messages[-1].content,
                k=k
            )

            if rag_pack and "system_prompt" in rag_pack and "user_prompt" in rag_pack:
                used_rag = True
                ctx = rag_pack.get("context", "")

                # Sem contexto real -> fallback direto
                if not ctx.strip():
                    gen_text = FALLBACK
                    end = time.time()
                    public_messages = _make_public_messages(original_user_question, gen_text)

                    DATABASE.add_operation(
                        user_name=inference_data.user_name,
                        system_response=gen_text,
                        inference_data=inference_data
                    )
                    return InferenceResponseDTO(
                        messages=public_messages,
                        generated_text=gen_text,
                        inference_time_in_seconds=end - start,
                    )

                # ✅ Quando há RAG, usamos LoRA
                model_to_use = LORA_MODEL
                inference_data.messages = [
                    MessageDTO(role="system", content=rag_pack["system_prompt"]),
                    MessageDTO(role="user", content=rag_pack["user_prompt"]),
                ]
            else:
                # k>0 mas não há contexto -> fallback direto
                gen_text = FALLBACK
                end = time.time()
                public_messages = _make_public_messages(original_user_question, gen_text)

                DATABASE.add_operation(
                    user_name=inference_data.user_name,
                    system_response=gen_text,
                    inference_data=inference_data
                )
                return InferenceResponseDTO(
                    messages=public_messages,
                    generated_text=gen_text,
                    inference_time_in_seconds=end - start,
                )

    except Exception as e:
        print(f"[RAG] disabled due to error: {e}")
        used_rag = False
        ctx = ""
        model_to_use = LLM_MODEL

    # -------------------------
    # 1.1) Se NÃO houve RAG (k==0), garante system default
    # -------------------------
    if not used_rag:
        if not inference_data.messages or inference_data.messages[0].role != "system":
            inference_data.messages = [
                MessageDTO(role="system", content=SYSTEM_DEFAULT),
                *inference_data.messages
            ]

    # -------------------------
    # 2) Inferência
    # -------------------------
    inference = await model_to_use.exec_inference(
        messages=inference_data.messages,
        response_num_tokens=inference_data.inference_parameters.tokens_count,
        response_temperature=inference_data.inference_parameters.temperature,
        top_k=inference_data.inference_parameters.top_k_tokens,
        top_p=inference_data.inference_parameters.top_p_tokens,
        do_sample=inference_data.inference_parameters.do_sample,
    )

    end = time.time()

    _, gen_text = process_generated_text(inference.split("</s>"))

    # limpar tags tipo FINAL ANSWER / ANSWER
    for tag in ("FINAL ANSWER:", "ANSWER:"):
        if tag in gen_text:
            gen_text = gen_text.split(tag, 1)[-1].strip()

    # -------------------------
    # 3) Guardrail (só quando houve RAG)
    # -------------------------
    if used_rag and ctx.strip():
        if not _is_unknown(gen_text) and gen_text.strip() != FALLBACK:
            keywords = [w.lower() for w in re.findall(r"\b[a-zA-Z]{6,}\b", ctx)][:12]
            if keywords and not any(k in gen_text.lower() for k in keywords):
                gen_text = FALLBACK

    # -------------------------
    # 4) DB + resposta
    # -------------------------
    DATABASE.add_operation(
        user_name=inference_data.user_name,
        system_response=gen_text,
        inference_data=inference_data
    )

    public_messages = _make_public_messages(original_user_question, gen_text)

    return InferenceResponseDTO(
        messages=public_messages,
        generated_text=gen_text,
        inference_time_in_seconds=end - start,
    )
