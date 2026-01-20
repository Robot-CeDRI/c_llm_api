from pydantic import BaseModel, conint, confloat

class InferenceParameters(BaseModel):
    temperature: confloat(ge=0.0, le=1.0) = 0.2
    tokens_count: conint(ge=1, le=2048) = 128
    top_k_tokens: conint(ge=0, le=100) = 50
    top_p_tokens: confloat(ge=0.0, le=1.0) = 0.95
    do_sample: bool = False