import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision.models import densenet121
from torchvision.models import efficientnet_v2_s
from torchvision.models import vgg13_bn

from typing import Tuple, List, Dict, Union, cast


class last_layer(nn.Module):
    def __init__(self, n_input: int, n_classes: int):
        super().__init__()
        self.mu  = nn.Linear(n_input, n_classes)
        self.tau = nn.Parameter(torch.ones([1, n_classes])) 

    def sigma(self, eps: float = 1e-6) -> Tensor:
        return F.softplus(self.tau) + eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.mu(x)
    

class last_layer_coeff_cdc(nn.Module):
    def __init__(self, n_input: int, n_classes: int = 4):
        super().__init__()
        self.coef = nn.Linear(n_input, n_classes)


    def rational_fct_cdc(self, coefficients: float) -> Tensor:
        """
        Functional form of CDC: sum_i=0^3 a_i x^-i
        """
        diameters = torch.Tensor([0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.31, 
            0.40, 0.50, 0.63, 0.80, 1.00]).reshape(1, -1, 1)
        exponents = torch.Tensor([0, -1, -2, -3])
        diameters_scaling = diameters.pow(exponents).to(
            coefficients.get_device())
        cdc = (diameters_scaling * coefficients.unsqueeze(-2)).sum(-1)
        assert cdc.shape == (coefficients.shape[0], diameters.shape[1])

        return cdc


    def forward(self, x: Tensor) -> Tensor:
        coef = self.coef(x)
        return self.rational_fct_cdc(coef)


def choose_linear_model(n_input: int, predict_coeff: bool) -> nn.Module:
    if predict_coeff:
        return last_layer_coeff_cdc(n_input, n_classes = 4)
    else:
        return last_layer(n_input, n_classes = 12)


class CNN(nn.Module):
    def __init__(
        self, features: nn.Module, n_classifier_input: int = 512, 
            num_classes: int = 12, init_weights: bool = True, 
            dropout: float = 0.5) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(n_classifier_input * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", 
                        	nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], in_channels = 1, 
    batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    "A1": [64, "M", 128, "M", 256, "M", 512, "M"],
    "A2": [32, "M", 64, "M", 128, "M", 256, "M"],
    "A3": [16, "M", 32, "M", 64, "M", 128, "M"],
    "B1": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M"],
    "B2": [32, 32, "M", 64, 64, "M", 128, 128, "M", 256, 256, "M"],
    "B3": [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "M"]
}

def cnnb3bn(in_channels = 1) -> CNN:
    return CNN(make_layers(cfgs["B3"], in_channels, batch_norm = True),
        n_classifier_input = 128)


def get_backbone(model_name: str):
    if model_name == "resnet18":
        backbone = resnet18(weights = None)
    elif model_name == "densenet121":
        backbone = densenet121(weights = None)
    elif model_name == "efficientnets":
        backbone = efficientnet_v2_s(weights = None)
    elif model_name == "vgg13bn":
        backbone = vgg13_bn(weights = None)
    elif model_name == "cnnb3bn":
        backbone = cnnb3bn(in_channels = 1)   
    else:
        raise NotImplementedError("This backbone is not implemented.")
    
    return backbone
