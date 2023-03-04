from torch import nn

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord: float=5.0, lambda_noobj: float=0.5):
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, x, y):
        