import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BaselineDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.df['decayed'] = self.df['decayed'].apply(lambda x: 1 if x == 'True' else 0)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name, __, __, label = self.df.iloc[idx]
        img_fname = f'../Dataset/sample_data/image/{img_name}'
        img = Image.open(img_fname)
        img = np.array(img)

        if self.transform:
            img = self.transform(img)
        
        return img, label
    
class BaselineTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df['filename'].tolist()
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data[idx]
        img_fname = f'../Dataset/sample_data/image/{img_name}'
        img = Image.open(img_fname)
        img = np.array(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img