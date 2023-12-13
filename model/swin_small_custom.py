import torchvision
import torch
import torch.nn as nn
import timm
from model.basemodel import BaseModel

class Swin_small(BaseModel):
    def __init__(self):
        super(Swin_small, self).__init__()

        model_name = 'swin_small_patch4_window7_224'
        pretrained_weights_path = '../weights/swin_small_patch4_window7_224-ms_in22k_ft_in1k_weights.pth'

        model = timm.create_model(model_name, pretrained=False)
        state_dict = torch.load(pretrained_weights_path)
        model.load_state_dict(state_dict)

        self.model = model
        n_features = self.model.head.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.model.head.fc = self.fc

    def forward(self, x):
        x = self.model(x)
        return x