#!/usr/bin/env python3
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from captum.insights import AttributionVisualizer, Batch
from captum.insights.features import ImageFeature


def get_classes():
    classes = [
        "Plane",
        "Car",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]
    return classes


def get_pretrained_model():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    pt_path = os.path.abspath(
        os.path.dirname(__file__) + "/models/cifar_torchvision.pt"
    )
    net.load_state_dict(torch.load(pt_path))
    return net


def baseline_func(input):
    return input * 0


def formatted_data_iter():
    dataset = torchvision.datasets.CIFAR10(
        root="data/test", train=False, download=True, transform=transforms.ToTensor()
    )
    dataloader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    )
    while True:
        images, labels = next(dataloader)
        yield Batch(inputs=images, labels=labels)


if __name__ == "__main__":
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    model = get_pretrained_model()
    visualizer = AttributionVisualizer(
        models=[model],
        score_func=lambda o: torch.nn.functional.softmax(o, 1),
        classes=get_classes(),
        features=[
            ImageFeature(
                "Photo",
                baseline_transforms=[baseline_func],
                input_transforms=[normalize],
            )
        ],
        dataset=formatted_data_iter(),
    )

    visualizer.serve(debug=True)
