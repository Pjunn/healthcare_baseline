import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


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