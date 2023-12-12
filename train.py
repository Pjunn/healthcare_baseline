import warnings
warnings.filterwarnings('ignore')
import glob
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import logging
import os
import shutil


from datasets.datasets import BaselineDataset, BaselineTestDataset
from trainer.trainer import *

import importlib
def convert_kst(utc_string):
    dt_tm_utc = datetime.strptime(utc_string,"%Y-%m-%d %H:%M:%S")
    tm_kst = dt_tm_utc + timedelta(hours = 9)
    str_datetime = tm_kst.strftime("%Y-%m-%d_%H:%M:%S")
    return str_datetime

def load_config(config_path):
    with open(config_path) as file:
        config = json.load(file)
    return config

def get_optimizer(optimizer_config, parameters):
    if optimizer_config['type'] == 'Adam':
        return optim.Adam(parameters, **optimizer_config['args'])
    elif optimizer_config['type'] == 'AdamW':
        return optim.AdamW(parameters, **optimizer_config['args'])
    elif optimizer_config['type'] == 'Lion':
        return optim.Lion(parameters, **optimizer_config['args'])
    elif optimizer_config['type'] == 'RMSprop':
        return optim.RMSprop(parameters, **optimizer_config['args']) #  lr=0.001, alpha=0.9, weight_decay=0.01
    # Add other optimizer types here if needed

def get_lr_scheduler(scheduler_config, optimizer):
    if scheduler_config['type'] == 'CosineAnnealingWarmupRestarts':
        from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        
        return CosineAnnealingWarmupRestarts(optimizer, **scheduler_config['args'])
    # Add other scheduler types here if needed
    
def get_model(model_name):
    # Convert the model name to lowercase to create the file name.
    model_file = model_name.lower()
    # Dynamically import the file.
    module = importlib.import_module(f"model.{model_file}_custom")
    # Retrieve the class from the module.
    model_class = getattr(module, model_name)
    return model_class

def create_model_filename(config):
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"{current_time}_{config['thismodel']}_{config['augmentation']}_{config['criterion']}_{config['size']}_{config['num_batches']}_{config['num_epochs']}.pth"
    return filename

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main(config_path):
    # get JSON config
    config = load_config(config_path)
    
    # set seed
    seed_everything(config['seed'])
    
    # make save file name from config
    filename = create_model_filename(config)

    # parameter for save
    min_val_loss = float('inf')

    early_stop_counter = 0
    early_stopping_epochs = config['early_stop']

    # Set logger
    if not os.path.exists('logs'):
        os.makedirs('logs')
    filename_wo_pth = filename.split('.')[0]
    f = open(f'./logs/{filename_wo_pth}.log', 'w')
    f.close
    logging.basicConfig(filename=f'./logs/{filename_wo_pth}.log',
                        filemode='a',
                        format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO,
                        datefmt='%m/%d/%Y %I:%M:%S %p',)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load hyperparameters and settings from config
    num_epochs = config['num_epochs']
    num_batches = config['num_batches']
    size = config['size']

    # Get model
    thismodel = get_model(config['thismodel'])

    # Set augmentation module
    transform_module = getattr(importlib.import_module("datasets.augmentation"), config['augmentation'])
    tr_transform = transform_module(img_size=size, is_train=True)
    val_transform = transform_module(img_size=size, is_train=False)
    
    # Set criterion
    criterion = config['criterion']

    # Load the data
    df = pd.read_csv(f'../Dataset/input.csv')
    train_df, val_df = train_test_split(df, test_size=0.25, stratify=df['teeth_num'], random_state=20231101)

    # Prepare dataset
    train_dataset = BaselineDataset(train_df, transform=tr_transform)
    val_dataset = BaselineDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=num_batches, shuffle=False, num_workers=config['num_workers'])

    # Initialize the model, loss function, optimizer, and lr_scheduler
    model = thismodel().to(device)
    optimizer = get_optimizer(config['optimizer'], model.parameters())
    lr_scheduler = get_lr_scheduler(config['lr_scheduler'], optimizer)

    # Train the model
    for epoch in range(num_epochs):
        if config['amp']:
            train_losses = train_amp(model, train_loader, criterion, optimizer, device)
        else:
            train_losses = train(model, train_loader, criterion, optimizer, device)
        val_losses, val_metrics, val_class_f1_scores, valid_accuracy = valid(model, val_loader, criterion, device)

        print('Epoch {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Metric: {:.4f}, Valid class_f1score:{}, Valid Accuracy: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(val_losses), np.mean(val_metrics),val_class_f1_scores), valid_accuracy)
        logging.info('Epoch {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Metric: {:.4f}, Valid class_f1score:{}, Valid Accuracy: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(val_losses), np.mean(val_metrics),val_class_f1_scores), valid_accuracy)
        
        # Update learning rate 
        lr_scheduler.step()

        avg_val_loss = np.mean(val_losses)

        val_metric_str = '{:05}'.format(int(val_metrics * 10000))
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss

            # Reset early stopping counter
            early_stop_counter = 0
            best_model_filename = filename.replace(".pth", f"_metric_{val_metric_str}_epoch{epoch+1}.pth")
            newfilename = filename.replace(".pth", "")

            folder_name = f"./runs/{newfilename}/epoch_{(epoch//5+1)*5}" # if epoch is 1 it will be 5
            os.makedirs(folder_name, exist_ok=True)

            best_model_path = os.path.join(folder_name, best_model_filename)

            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1

        # Every 5 epochs, check if the best model of this block is better than the last saved model
        if (epoch + 1) % 5 == 0:
            newfilename = filename.replace(".pth", "")
            folder_name = f"./runs/{newfilename}/epoch_{epoch+1}"
            
            # Delete all .pth files except best model
            if os.path.exists(folder_name):
                for model_filename in os.listdir(folder_name):
                    if model_filename != best_model_filename:
                        os.remove(os.path.join(folder_name, model_filename))

            # Print minimum validation loss for 5 epochs
            print(f'Current best loss: {best_model_path}')
            logging.info('Epoch {},Current best loss: {}'.format(epoch+1,best_model_path))

        if early_stopping_epochs == early_stop_counter:
            break

    print(f'Final best loss: {best_model_path}')
    logging.info(f'Final best loss: {best_model_path}')

    print(f'Final Please type python predict.py --pth {best_model_path}')
    logging.info(f'Final Please type python predict.py --pth {best_model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)