from pydantic import BaseModel, conint, confloat

class InferenceParameters(BaseModel):
    temperature: confloat(ge=0.0, le=1.0)
    tokens_count: int
    top_k_tokens: conint(ge=0, le=100)
    top_p_tokens: confloat(ge=0.0, le=1.0)
    do_sample: bool