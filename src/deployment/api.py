from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn

from src.utils.handle_deployment import handle_input_inference, handle_output_inference, load_model, make_prediction

app = FastAPI()

@app.get('/')
async def root() -> RedirectResponse:
    """
    Root of the api, redirects user to the /docs page

    Returns:
        RedirectResponse: Redirects user to the /docs page
    """
    return RedirectResponse(url="/docs")

@app.post("/predict/")
async def predict_segmentation(file: UploadFile) -> FileResponse:
    """
    This function adds a post to the api where the user can use the model to predict a segmentation map of an image

    Args:
        file (UploadFile): file that has been uploaded by user

    Returns:
        FileResponse: File that will be returned to the user, 
        which will be a zip file, which contains the original images and the predictions made by the model.
    """ 
    input_path, temp_input_dir, temp_output_dir = handle_input_inference(file)
        
    model = load_model()
    make_prediction(model, input_path, temp_output_dir)
    
    zip_path = handle_output_inference(temp_input_dir, temp_output_dir)
    return FileResponse(path=zip_path, media_type='application/zip', filename='output.zip')

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)