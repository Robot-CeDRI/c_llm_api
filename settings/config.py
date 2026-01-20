import os


class Common(object):
    CUR_VERSION = "1.1.0"
    API_HOST = "localhost"
    API_PORT = 8080

    MODELS_DIR = "Models"
    HF_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    RAW_MODEL_PATH = "Models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
    RAG_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
    # LoRA fine-tuned adapter (SFT)
    LORA_ADAPTER_DIR = "Experiments/outputs/tinyllama-ipb-sft-lora"


    FINE_TUNED_MODEL_PATH = "" #"/home/cedri/Desktop/git_automations/Repositories/c_cedrinho_llm_model/model--CedrinhoLLM"
    DATABASE_FILE = "API_Database.db"
    MAX_RAG_DISTANCE = 1.05

    MAX_CACHE_CONVERSATIONS = 500
    LLM_BATCH_SIZE = 1          # para 1 request (FASTAPI), 1 é melhor
    LLM_MAX_NEW_TOKENS_DEFAULT = 80
    # GPU perf flags (seguro)
    LLM_USE_TF32 = True

class Dev(Common):
    HF_TOKEN = "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"
    DEBUG = True
    # DEV = mais rápido no teu PC
    LLM_BATCH_SIZE = 1
    LLM_USE_TF32 = True
    LLM_ATTN_IMPL = None
    DEFAULT_RAG_K = 3
    MIN_RAG_SCORE = 0.35


class Production(Common):
    HF_TOKEN = "CEDRI-HF-TOKEN"
    DEBUG = False
    # ROBOT/PROD = conservador
    LLM_BATCH_SIZE = 1
    LLM_USE_TF32 = False
    LLM_ATTN_IMPL = None
    DEFAULT_RAG_K = 3
    MIN_RAG_SCORE = 0.35


class Staging(Production):
    DEBUG = True
