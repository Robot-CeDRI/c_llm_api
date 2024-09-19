from settings import auto_config as cfg

from fastapi import FastAPI
from starlette.responses import RedirectResponse

import API.Controllers.ApiInfo_Controller as APIInfoController
import API.Controllers.Inferences_controller as InferenceController


app = FastAPI()

app.include_router(APIInfoController.router, prefix='/info', tags=['Info'])
app.include_router(InferenceController.router, prefix='/inferences', tags=['Inferences'])

@app.get("/", tags=["Default"])
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg.API_HOST, port=int(cfg.API_PORT))