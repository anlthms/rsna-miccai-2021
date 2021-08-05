from torch import nn
from efficientnet_pytorch_3d import EfficientNet3D

class Model(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.net = EfficientNet3D.from_name(conf.arch, override_params={'num_classes': 2}, in_channels=1)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        out = self.net(x)
        return out

