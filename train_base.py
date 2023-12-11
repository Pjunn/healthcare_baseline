import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score


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


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()

        self.model = torchvision.models.densenet121(pretrained=True)
        n_features = self.model.classifier.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.model.classifier = self.fc

    def forward(self, x):
        x = self.model(x)
        return x


# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.float().to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs).view(-1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


# Define the validation function
def valid(model, val_loader, criterion, device):
    model.eval()
    losses, metrics = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs).view(-1)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            preds = torch.sigmoid(outputs).round()
            metrics.append(f1_score(labels.cpu(), preds.cpu(), average='macro'))
    return losses, metrics


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 20
num_batches = 32

# transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the data
df = pd.read_csv(f'/DATA/train/train.csv')

# train / validation split with stratified sampling
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20231101)

for train_idx, val_idx in skf.split(df, df['risk']):
    # use first fold as validation set
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    break

# prepare dataset
train_dataset = BaselineDataset(train_df, transform=train_transform)
val_dataset = BaselineDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=num_batches, shuffle=False)

# Initialize the model, loss function, and optimizer
model = BaselineModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Train the model
train_loss_list, val_loss_list, val_metric_list = [], [], []
for epoch in range(num_epochs):
    train_losses = train(model, train_loader, criterion, optimizer, device)
    #val_losses, val_metrics = valid(model, val_loader, criterion, device)
    print('Epoch {}, Train Loss: {:.4f}'.format(epoch+1, np.mean(train_losses)))
    #print('Epoch {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Metric: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(val_losses), np.mean(val_metrics)))

# save model
torch.save(model.state_dict(), '/USER/model.pth')
