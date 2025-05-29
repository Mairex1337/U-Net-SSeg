import argparse
import base64
import json
import os


def convert_images_to_json(path_to_images: str, output_path: str, file_name: str = "images.json") -> None:
    """
    Converts a folder of image(s) into a byte array JSON file, that can be used for segmentation prediction

    Args:
        path_to_images (str): Path where the images are stored.
        output_path (str): Path where the JSON will be saved.
        file_name (str): File name for the .json.
    """     
    dict_imgs = {
        'image_names': [],
        'images': []
    }
    
    images = os.listdir(path_to_images)
    for img in images:
        with open(os.path.join(path_to_images, img), "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        dict_imgs['image_names'].append(img)
        dict_imgs['images'].append(encoded_string)
    
    json_object = json.dumps(dict_imgs)
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, file_name)
    
    with open(json_path, "w") as outfile:
        outfile.write(json_object)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert images to JSON")
    parser.add_argument('--path-to-images', required=True, type=str)
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--file-name', type=str)
    args = parser.parse_args()
    if args.file_name:
        convert_images_to_json(args.path_to_images, args.output_path, args.file_name)
    else:
        convert_images_to_json(args.path_to_images, args.output_path)