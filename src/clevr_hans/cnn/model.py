import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34Small(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34Small, self).__init__()
        original_model = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
