import sys
sys.path.append(r'C:\Users\daand\RUG\applied ml\project\U-Net-SSeg')
from fastapi import FastAPI
import uvicorn

from src.utils.load_model import load_model
from src.utils.make_prediction import make_prediction

app = FastAPI()


@app.post("/predict")
async def predict_segmentation():
    model = load_model()
    img_dir = 'inference_input/'
    make_prediction(model, img_dir)
    return 'succesfull'

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)