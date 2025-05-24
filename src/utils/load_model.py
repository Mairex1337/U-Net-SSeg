import torch
from src.utils import read_config, get_device, get_model, get_run_dir

def load_model():
    
    #model_path needs to be in config when model is saved
    model_path = r'C:\Users\daand\RUG\applied ml\project\U-Net-SSeg\outputs\baseline\2\checkpoints\chkpt_epoch_23_best.pth'
    model = torch.load(model_path)
    model.eval()
    
    return model

def preprocess(input):
    input_tensor = torch.tensor([input])
    return input_tensor

def make_prediction(model , input):
    input_tensor = preprocess(input)
    with torch.no_grad():
        prediction = model(input_tensor).item()
        
    return prediction