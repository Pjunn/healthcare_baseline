import torch.nn as nn
import torchvision

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