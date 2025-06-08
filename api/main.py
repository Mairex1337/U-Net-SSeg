import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.routes import predict, predict_image
from api.utils import cleanup_temp_dirs
from src.utils import resolve_path

app = FastAPI(
    title="ML Inference API",
    description="Upload a road scene image and receive a segmentation mask from a trained u-net model.",
    version="1.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(predict_image.router)


@app.get('/', include_in_schema=False)
async def root() -> RedirectResponse:
    """
    Root of the api, redirects user to the /docs page

    Returns:
        RedirectResponse: Redirects user to the /docs page
    """
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    cleanup_temp_dirs(resolve_path(""))
    uvicorn.run(app, host="127.0.0.1", port=8000)
