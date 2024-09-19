from fastapi import APIRouter
from settings import auto_config as cfg

from API.DataTransferObjects.Responses.apiInfoResponseDTO import APIInfoDTO
from API.Utils.Setup_LLM import LLM_MODEL

router = APIRouter()

@router.get("/", summary="Get API running information.", response_model=APIInfoDTO)
async def api_info():
    return APIInfoDTO(
        version=cfg.CUR_VERSION,
        description="This is a Large Language Model API for text generation.",
        base_model=cfg.HF_LLM_MODEL,
        fine_tuned_model=cfg.FINE_TUNED_MODEL_PATH,
        device=LLM_MODEL.device.__repr__()
    )