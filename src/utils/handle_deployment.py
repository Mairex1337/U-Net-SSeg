import os
import shutil

from fastapi import UploadFile
from PIL import Image

def handle_input_inference(file: UploadFile):
    cwd = os.getcwd()
    
    input_folder_temp = 'temp_input/'
    output_folder_temp = 'temp_output/'
    os.mkdir(input_folder_temp)
    os.mkdir(output_folder_temp)
    
    if file.filename.endswith('.jpg'):
        im = Image.open(file.file)
        im.save(os.path.join(cwd, input_folder_temp, file.filename),'JPEG')
        input_path = input_folder_temp        
    elif file.filename.endswith('.zip'):
        zip_path_input = os.path.join(input_folder_temp, file.filename)
        
        with open(zip_path_input, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        shutil.unpack_archive(zip_path_input, input_folder_temp, 'zip')
        input_path =  zip_path_input.replace('.zip', '')
        
    return input_path, input_folder_temp, output_folder_temp

def handle_output_inference(temp_input_dir: str, temp_output_dir: str) -> str:
    cwd = os.getcwd()
    zip_name_output = 'output'
    
    shutil.make_archive(base_name=zip_name_output, format='zip', root_dir=cwd, base_dir=temp_output_dir)
    shutil.rmtree(os.path.join(cwd, temp_input_dir))
    shutil.rmtree(os.path.join(cwd, temp_output_dir))
    
    zip_path = os.path.join(cwd, zip_name_output + '.zip') 
    return zip_path