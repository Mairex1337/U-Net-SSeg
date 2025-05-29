import argparse
import base64
import json
import os


def convert_json_to_images(path_to_json: str, output_path: str) -> None:
    """
    Converts a JSON file containing base64-encoded images, masks, and colormaps into image files.
    Saves the images in the given output path.

    Expects JSON keys:
    - "images": list of base64-encoded input images.
    - "pred_mask": list of base64-encoded predicted mask images (grayscale).
    - "pred_color": list of base64-encoded predicted color mask images (colormap).

    Args:
        path_to_json (str): Path where the JSON is stored.
        output_path (str): Path where the images will be saved.
    """
    with open(path_to_json, 'r') as f:
        data = json.load(f)

    required_keys = ["images", "pred_mask", "pred_color"]
    assert all(key in data for key in required_keys), f"JSON must contain: {', '.join(required_keys)}"
    os.makedirs(output_path, exist_ok=True)

    names = data.get("image_names", [f"image_{i}.jpg" for i in range(len(data["images"]))])

    for i, name in enumerate(names):
        base_name, _ = os.path.splitext(name)

        # Decode and save input image
        input_bytes = base64.b64decode(data["images"][i])
        input_path = os.path.join(output_path, f"input_{base_name}.jpg")
        with open(input_path, "wb") as f:
            f.write(input_bytes)

        # Decode and save predicted mask
        mask_bytes = base64.b64decode(data["pred_mask"][i])
        mask_path = os.path.join(output_path, f"mask_{base_name}.png")
        with open(mask_path, "wb") as f:
            f.write(mask_bytes)

        # Decode and save colormapped prediction
        color_bytes = base64.b64decode(data["pred_color"][i])
        color_path = os.path.join(output_path, f"color_{base_name}.png")
        with open(color_path, "wb") as f:
            f.write(color_bytes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert JSON to images")
    parser.add_argument('--path-to-json', required=True, type=str)
    parser.add_argument('--output-path', required=True, type=str)
    args = parser.parse_args()

    convert_json_to_images(args.path_to_json, args.output_path)