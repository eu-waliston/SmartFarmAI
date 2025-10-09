import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AdvancedPlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(AdvancedPlantDiseaseCNN, self).__init__()

        # Usando ResNet50 pré-treinado com backbone
        self.backbone = models.resnet50(pretrained=use_pretrained)

        # Congelando as primeira camadas
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Substituindo a camada final
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self,x):
        return self.backbone(x)

class MultiModalPlantModel(nn.Module):
    def __init__(self, num_image_classes, num_sensor_features):
        super(MultiModalPlantModel, self).__init__()

        # Branch de imagem
        self.sensor_branch = nn.Sequential(
            nn.Linear(num_sensor_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        # Fusão multimodal
        self.classifier = nn.Sequential(
            nn.Linear(num_image_classes + 32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        # Fusão multimodal
        self.classifier = nn.Sequential(
            nn.Linear(num_image_classes + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_image_classes)
        )

    def forward(self, images, sensor_data):
        image_features = self.image_branch(images)
        senosr_features = self.sensor_branch(sensor_data)

        combined = torch.cat([image_features, senosr_features], dim=1)
        output = self.classifier(combined)

        return output