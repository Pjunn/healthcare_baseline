import torchvision
import torch
import torch.nn as nn
import timm
from model.basemodel import BaseModel

class Efficientnet_v2_small(BaseModel):
    def __init__(self):
        super(Efficientnet_v2_small, self).__init__()

        model_name = 'tf_efficientnetv2_s'
        pretrained_weights_path = '../weights/tf_efficientnetv2_s-in21k_ft_in1k_weights.pth'
        # 수정바람

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