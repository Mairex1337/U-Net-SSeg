# Very Epic Machine Learning Project

`WOW`

### How to use model for inference
- run pip install -r requirements.txt to install all libraries
- run python -m src.deployment.api to start the api
  
Under /predict/, you can upload the files you want for segmentation prediction.
This can either be a single image (JPG) or a ZIP file containing multiple images (JPG).
For the latter, the ZIP file should contain only JPG images, no directories or other file types.
The model will return a ZIP folder that includes a temp_output folder, which contains a folder with the original images
and a folder with the segmentation predictions.

### Rules for running files and imports so that it consistently works!!!
- Always use absolute imports -> `from src.data.dataset import SegmentationDataset`
- Always run files from the root of the directory!
- Run them like this and it will always work -> `python -m src.utils.calculate_means` (no `.py`)
- If you want your VSCode play button to work, set this in `.vscode/launch.json` in the root of the project:
    `"cwd": "${workspaceFolder}"`

### How to set up your data/ directory
- Make sure you downloaded the zips for semantic segmentation, 10kimages train, 10kimages val, 10kimages test.
- Order them as specified in the cfg.yaml.

### Other
- Please also install and configure Isort plugin to sort imports nicely :)
    - Put this into your settings.json after installing:
```json
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit",
    }
  }
```
- Hyperparameters should be stored in cfg.yaml in the future, there is a read_config function in utils/ dir.