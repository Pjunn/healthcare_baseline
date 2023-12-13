import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

from datasets.datasets import *
from datasets.augmentation import *
from model.basemodel import *
from importlib import import_module

def load_model(model_name, saved_model, device):
    
    model_module_name = "model." + model_name.lower() + "_custom"
    model_module = getattr(import_module(model_module_name), model_name)
    model = model_module().to(device)

    model.load_state_dict(torch.load(saved_model, map_location=device))

    return model

def restore_config(model_path):
    model_dir = model_path.split('/')[2]
    model_config_list = model_dir.split('_')[2:]
    config = dict()
    config['num_epochs'] = int(model_config_list.pop())
    config['num_batches'] = int(model_config_list.pop())
    config['size'] = int(model_config_list.pop())
    config['criterion'] = model_config_list.pop()
    config['augmentation'] = model_config_list.pop()
    config['model'] = '_'.join(model_config_list)
    return config

def main(model_path):
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = restore_config(model_path)
    # image size setting
    size = config['size']

    # transformations 
    augmentation = config['augmentation']
    test_transform = augmentation(img_size=size, is_train=False)

    # hyperparameters
    num_batches = config['num_batches']

    # test_df = pd.read_csv(f'/DATA/sample/sample.csv')
    test_df = pd.read_csv(f'/DATA/test/test.csv') #TODO 데이터 경로 수정
    test_dataset = BaselineTestDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=num_batches, shuffle=False, num_workers=4)

    # model setting
    model = load_model(config['model'], model_path, device)

    model.eval()
    preds_list = []
    with torch.no_grad():
        for image in tqdm(test_loader):
            image = image.to(device)
            outputs = model(image).view(-1)

            preds = torch.sigmoid(outputs).round()
            preds_list += preds.cpu().numpy().tolist()

    test_df['decayed'] = preds_list
    test_df['decayed'] = test_df['decayed'].apply(lambda x: 'true' if x == 1 else 'false')
    
    # Create a folder for results if it doesn't exist
    os.makedirs('result', exist_ok=True)

    # Create the filename by changing the extension of the model filename to .csv
    result_filename = os.path.join('result', os.path.basename(model_path).replace('.pth', '.csv'))
    test_df.to_csv(result_filename, index=False)
    print(f'Please type cat ./{result_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, required=True, help='Path to the model pth file')
    args = parser.parse_args()
    main(args.pth)