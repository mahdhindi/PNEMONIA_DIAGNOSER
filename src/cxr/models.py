"""The five algorithms compared.  All emit one logit (P(pneumonia))."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class MajorityClass(nn.Module):
    """Trivial reference: always predict the training-set majority class.
    Has no trainable parameters; `fit` stores the training prior as a logit."""

    def __init__(self):
        super().__init__()
        self.register_buffer("logit", torch.zeros(1))

    def fit(self, prior_pos: float):
        p = min(max(prior_pos, 1e-6), 1 - 1e-6)
        self.logit.fill_(float(torch.logit(torch.tensor(p))))

    def forward(self, x):
        return self.logit.expand(x.shape[0], 1)


class LogisticRegression(nn.Module):
    """The required simple baseline: a single linear layer on the flattened,
    normalised pixels (= multinomial/binary logistic regression trained by
    gradient descent).  Permutation-invariant w.r.t. pixel positions."""

    def __init__(self, img_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * img_size * img_size, 1))

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Compact CNN trained from scratch (4 conv blocks, ~0.4M params)."""

    def __init__(self, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, c_in = [], 3
        for c in widths:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, 1))

    def forward(self, x):
        return self.head(self.features(x))


def resnet50(pretrained: bool = True):
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = torchvision.models.resnet50(weights=w)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def densenet121(pretrained: bool = True):
    w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.densenet121(weights=w)
    m.classifier = nn.Linear(m.classifier.in_features, 1)
    return m


# name -> (constructor, default learning rate, needs training?)
ZOO = {
    "majority":    dict(lr=0.0,  trainable=False, family="baseline (no learning)"),
    "logreg":      dict(lr=1e-4, trainable=True,  family="linear baseline"),
    "smallcnn":    dict(lr=1e-3, trainable=True,  family="CNN from scratch"),
    "resnet50":    dict(lr=1e-4, trainable=True,  family="CNN, ImageNet-pretrained, fine-tuned"),
    "densenet121": dict(lr=1e-4, trainable=True,  family="CNN, ImageNet-pretrained, fine-tuned"),
}


def build_model(name: str, img_size: int, pretrained: bool = True) -> nn.Module:
    if name == "majority":
        return MajorityClass()
    if name == "logreg":
        return LogisticRegression(img_size)
    if name == "smallcnn":
        return SmallCNN()
    if name == "resnet50":
        return resnet50(pretrained)
    if name == "densenet121":
        return densenet121(pretrained)
    raise ValueError(f"unknown model {name}; choose from {list(ZOO)}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
