import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from trainer.loss_custom import *

# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = []
    criterion = create_criterion(criterion)
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
    losses = []
    all_labels, all_preds = [], []
    criterion = create_criterion(criterion)
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs).view(-1)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds = torch.sigmoid(outputs).round()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    metrics = f1_score(all_labels, all_preds, average='macro')
    class_f1_scores = f1_score(all_labels, all_preds, average=None)
    accuracy = accuracy_score(all_labels, all_preds)
    return losses, metrics, class_f1_scores, accuracy