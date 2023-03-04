from typing import List

import torch
from torch import nn

class YOLOv1(nn.Module):
    def __init__(self, num_boxes=2, num_classes=20):
        super(YOLOv1, self).__init__()

        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.model = nn.Sequential(
            BasicBlock(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicBlock(64, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            *create_yolo_block(192, [128, 256, 256, 512]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            *create_yolo_block(512, [256, 512] * 4 + [512, 1024]),
            nn.MaxPool2d(kernel_size=2, stride=2),

            *create_yolo_block(1024, [512, 1024] * 2),
            BasicBlock(1024, 1024, kernel_size=3),
            BasicBlock(1024, 1024, kernel_size=3, stride=2),
            
            BasicBlock(1024, 1024, kernel_size=3),
            BasicBlock(1024, 1024, kernel_size=3),

            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),

            nn.Linear(4096, 7 * 7 * (num_boxes * 5 + num_classes))
        )
    
    def forward(self, x):
        x = self.model(x)
        return x.reshape(-1, 7, 7, self.num_boxes * 5 + self.num_classes)

class BasicBlock(nn.Module):
    """
    Basic YOLO Conv, BatchNorm, LeakyReLU block.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.block(x)

def create_yolo_block(in_channels: int, num_filters: List) -> List[nn.Module]:
    """Create a YOLO block that alternates 1x1 and 3x3 conv layers.

    Args:
        in_channels (int): Number of incoming channels in the first layer
        num_filters (List): List of number of filters for each layer.
    Returns:
        List[nn.Module]: List of layers.
    """
    layers = []
    num_filters = [in_channels] + num_filters
    for i, in_channels in enumerate(num_filters[:-1]):
        out_channels = num_filters[i+1]
        kernel_size = 3 if (i + 1) // 2 else 1  # 1 for even layers, 3 for odd
        layers.append(
            BasicBlock(in_channels, out_channels, kernel_size=kernel_size)
        )
        return layers