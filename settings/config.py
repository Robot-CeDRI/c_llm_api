import os


class Common(object):
    CUR_VERSION = "1.0.1"
    API_HOST = "localhost"
    API_PORT = 8080
    MODELS_DIR = "Models"
    HF_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    RAW_MODEL_PATH = "Models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
    FINE_TUNED_MODEL_PATH = "Models/Current-Cedrinho-API-Model"
    CACHE_CON_STRING = "sqlite:///conversation_cache.db"
    MAX_CACHE_CONVERSATIONS = 500

class Dev(Common):
    HF_TOKEN = "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"
    DEBUG = True


class Production(Common):
    HF_TOKEN = "CEDRI-HF-TOKEN"
    DEBUG = False


class Staging(Production):
    DEBUG = True
