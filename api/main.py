import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.utils import resolve_path
from api.routes import predict, json_img, img_json
from api.utils import cleanup_temp_dirs


app = FastAPI(
    title="ML Inference API",
    description="Upload an image to receive a segmentation map from the ML model.",
    version="1.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(json_img.router, tags=["JSON to Images"])
app.include_router(img_json.router, tags=["Images to JSON"])

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
