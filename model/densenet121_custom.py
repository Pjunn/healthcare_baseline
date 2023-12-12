import torchvision
import torch
import torch.nn as nn
import timm
from model.basemodel import BaseModel

class Densenet121(BaseModel):
    def __init__(self):
        super(Densenet121, self).__init__()

        model_name = 'densenet121'
        pretrained_weights_path = '/USER/weights/densenet121_weights.pth'

        model = timm.create_model(model_name, pretrained=False)
        state_dict = torch.load(pretrained_weights_path)
        model.load_state_dict(state_dict)

        self.model = model
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