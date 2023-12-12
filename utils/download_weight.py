import timm
import torch

def download(model_name):

    model = timm.create_model(model_name, pretrained=True)
    
    state_dict = model.state_dict()

    model_name = model_name.replace('.','-')

    weights_path = f'{model_name}_weights.pth'

    torch.save(state_dict, weights_path)

    print(f"{model_name} model saved at: {weights_path}")
