import sys
sys.path.append(r'C:\Users\daand\RUG\applied ml\project\U-Net-SSeg')

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn

from src.utils.load_model import load_model
from src.utils.make_prediction import make_prediction

app = FastAPI()

@app.get('/')
async def root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict_segmentation():
    model = load_model()
    img_dir = 'inference_input/'
    make_prediction(model, img_dir)
    out_dir = 'inference_output/outputs/predictions/00000.png'
    cwd = os.getcwd()
    full_path = os.path.join(cwd, out_dir)
    print(full_path)
    return FileResponse(full_path, media_type="application/octet-stream", filename=out_dir)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)