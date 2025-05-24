from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

def load_model():
    pass

def make_pred():
    pass

class User(BaseModel):
    pass

app = FastAPI()

@app.post("/file")
async def create_file(file: UploadFile = File(...)):
    return {"file_name": file.name}