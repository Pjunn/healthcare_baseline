from torch.utils.data import Dataset, DataLoader
from PIL import Image


class BaselineDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.df['risk'] = self.df['risk'].apply(lambda x: 1 if x == 'high' else 0)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name, label = self.df.iloc[idx]
        img_fname = f'/DATA/train/images/{img_name}'
        img = Image.open(img_fname)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label