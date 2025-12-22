import os


class Common(object):
    CUR_VERSION = "1.1.0"
    API_HOST = "localhost"
    API_PORT = 8080

    MODELS_DIR = "Models"
    HF_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    RAW_MODEL_PATH = "Models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
    RAG_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

    FINE_TUNED_MODEL_PATH = "" #"/home/cedri/Desktop/git_automations/Repositories/c_cedrinho_llm_model/model--CedrinhoLLM"
    DATABASE_FILE = "API_Database.db"
    MAX_RAG_DISTANCE = 1.05

    MAX_CACHE_CONVERSATIONS = 500

class Dev(Common):
    HF_TOKEN = "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"
    DEBUG = True


class Production(Common):
    HF_TOKEN = "CEDRI-HF-TOKEN"
    DEBUG = False


class Staging(Production):
    DEBUG = True
