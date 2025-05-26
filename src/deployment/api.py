import sys
sys.path.append(r'C:\Users\daand\RUG\applied ml\project\U-Net-SSeg')

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn

from src.utils.load_model import load_model
from src.utils.make_prediction import make_prediction
from src.utils.handle_deployment import handle_input_inference, handle_output_inference

app = FastAPI()

@app.get('/')
async def root():
    return RedirectResponse(url="/docs")

@app.post("/predict/")
async def predict_segmentation(file: UploadFile) -> FileResponse:    
    input_path, temp_input_dir, temp_output_dir = handle_input_inference(file)
        
    model = load_model()
    make_prediction(model, input_path, temp_output_dir)
    
    zip_path = handle_output_inference(temp_input_dir, temp_output_dir)
    return FileResponse(path=zip_path, media_type='application/zip', filename='output.zip')

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)