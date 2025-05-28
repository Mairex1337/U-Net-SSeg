import sys
sys.path.append(r'C:\Users\daand\RUG\applied ml\project\U-Net-SSeg')
import os
import shutil

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn

from src.utils.handle_deployment import handle_input_inference, handle_output_inference, load_model, make_prediction

app = FastAPI(swagger_ui_parameters={"defaultModelsExpandDepth": -1})

@app.get('/', include_in_schema=False)
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
    temp_input_dir, temp_output_dir = handle_input_inference(file)
    
    model = load_model()
    make_prediction(model, temp_input_dir, temp_output_dir)
    json_path = handle_output_inference(temp_input_dir, temp_output_dir)
    return FileResponse(path=json_path, media_type='application/json', filename='output.json')

if __name__ == "__main__":
    cwd = os.getcwd()
    output_zip_file = os.path.join(cwd, 'output.json')
    if os.path.isfile(output_zip_file):
        os.remove(output_zip_file)
        
    temp_input_dir = os.path.join(cwd, 'temp_input')
    if os.path.isdir(temp_input_dir):
        shutil.rmtree(temp_input_dir)
    
    temp_output_dir = os.path.join(cwd, 'temp_output')
    if os.path.isdir(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)