import argparse
import base64
import json
import os


def convert_json_to_images(path_to_json: str, output_path: str) -> None:
    """
    Converts a JSON file containing base64-encoded images, masks, and colormaps into image files.
    Saves the images in separate subdirectories within the given output path.

    Subfolders:
    - input/: original input images
    - mask/: predicted grayscale masks
    - color/: colormapped predicted masks

    Args:
        path_to_json (str): Path where the JSON is stored.
        output_path (str): Path where the subfolders and images will be saved.
    """
    with open(path_to_json, 'r') as f:
        data = json.load(f)

    required_keys = ["images", "pred_mask", "pred_color"]
    assert all(key in data for key in required_keys), f"JSON must contain: {', '.join(required_keys)}"

    # Create subdirectories
    input_dir = os.path.join(output_path, "input")
    mask_dir = os.path.join(output_path, "mask")
    color_dir = os.path.join(output_path, "color")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    names = data.get("image_names", [f"image_{i}.jpg" for i in range(len(data["images"]))])

    for i, name in enumerate(names):
        base_name, _ = os.path.splitext(name)

        # Decode and save input image
        input_bytes = base64.b64decode(data["images"][i])
        with open(os.path.join(input_dir, f"{base_name}.jpg"), "wb") as f:
            f.write(input_bytes)

        # Decode and save predicted mask
        mask_bytes = base64.b64decode(data["pred_mask"][i])
        with open(os.path.join(mask_dir, f"{base_name}.png"), "wb") as f:
            f.write(mask_bytes)

        # Decode and save colormapped prediction
        color_bytes = base64.b64decode(data["pred_color"][i])
        with open(os.path.join(color_dir, f"{base_name}.png"), "wb") as f:
            f.write(color_bytes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert JSON to images in subdirectories")
    parser.add_argument('--path-to-json', required=True, type=str)
    parser.add_argument('--output-path', required=True, type=str)
    args = parser.parse_args()

    convert_json_to_images(args.path_to_json, args.output_path)
