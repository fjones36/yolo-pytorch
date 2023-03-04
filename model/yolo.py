from typing import List

import torch
from torch import nn

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *create_yolo_block(192, [128, 256, 256, 512]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *create_yolo_block(512, [256, 512] * 4 + [512, 1024]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *create_yolo_block(1024, [512, 1024] * 2),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.Conv2d(1024, 1024, kernel_size=3),
        )

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
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        )
        return layers