import os
import sys
import gdown
import zipfile
import shutil

def download_data(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def move_data(src, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dest_path = os.path.join(dest, item)
        if os.path.isdir(src_path):
            shutil.move(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    
    link = "https://drive.google.com/file/d/1rQHFQ5VVmkB07028O3Qvu9KGqk67hdOh/view?usp=drive_link"
    zip_file = "data.zip"
    temp_extract_path = "temp_data"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    try:
        download_data(link, zip_file)
        extract_zip(zip_file, temp_extract_path)
        move_data(temp_extract_path, os.path.join(project_root, "data"))
    finally:
        if os.path.exists(zip_file):
            os.remove(zip_file)
        if os.path.exists(temp_extract_path):
            shutil.rmtree(temp_extract_path)
        