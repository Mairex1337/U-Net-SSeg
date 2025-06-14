import os
import zipfile

import gdown


def download_data(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)
    print("Data has been successfully downloaded.")
def extract_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    print(f"Data has been successfully extracted into {zip_path}.")



if __name__ == "__main__":
    
    link = "https://drive.google.com/uc?id=1rQHFQ5VVmkB07028O3Qvu9KGqk67hdOh"
    zip_file = "data.zip"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    download_data(link, zip_file)
    extract_zip(zip_file)
    if os.path.exists(zip_file):
        os.remove(zip_file)
    print("Zip file has been deleted.")