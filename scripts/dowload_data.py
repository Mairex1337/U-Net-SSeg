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
        if os.path.isdir(src):
            shutil.move(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)