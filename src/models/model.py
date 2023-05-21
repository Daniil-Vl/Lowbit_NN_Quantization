import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights, googlenet

__all__ = [
    "ModelM3",
    "ModelM5",
    "ModelM7",
    "ResNet",
]


class ModelM3(nn.Module):
    def __init__(self):
        super(ModelM3, self).__init__()
        # output becomes 26x26
        self.conv1 = nn.Conv2d(1, 32, 3, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        # output becomes 24x24
        self.conv2 = nn.Conv2d(32, 48, 3, bias=False)
        self.conv2_bn = nn.BatchNorm2d(48)
        # output becomes 22x22
        self.conv3 = nn.Conv2d(48, 64, 3, bias=False)
        self.conv3_bn = nn.BatchNorm2d(64)
        # output becomes 20x20
        self.conv4 = nn.Conv2d(64, 80, 3, bias=False)
        self.conv4_bn = nn.BatchNorm2d(80)
        # output becomes 18x18
        self.conv5 = nn.Conv2d(80, 96, 3, bias=False)
        self.conv5_bn = nn.BatchNorm2d(96)
        # output becomes 16x16
        self.conv6 = nn.Conv2d(96, 112, 3, bias=False)
        self.conv6_bn = nn.BatchNorm2d(112)
        # output becomes 14x14
        self.conv7 = nn.Conv2d(112, 128, 3, bias=False)
        self.conv7_bn = nn.BatchNorm2d(128)
        # output becomes 12x12
        self.conv8 = nn.Conv2d(128, 144, 3, bias=False)
        self.conv8_bn = nn.BatchNorm2d(144)
        # output becomes 10x10
        self.conv9 = nn.Conv2d(144, 160, 3, bias=False)
        self.conv9_bn = nn.BatchNorm2d(160)
        self.conv10 = nn.Conv2d(160, 176, 3, bias=False)   # output becomes 8x8
        self.conv10_bn = nn.BatchNorm2d(176)
        self.fc1 = nn.Linear(11264, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)

    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))
        conv8 = F.relu(self.conv8_bn(self.conv8(conv7)))
        conv9 = F.relu(self.conv9_bn(self.conv9(conv8)))
        conv10 = F.relu(self.conv10_bn(self.conv10(conv9)))
        flat1 = torch.flatten(conv10.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits

    def forward(self, x):
        logits = self.get_logits(x)
        return logits
        # return F.log_softmax(logits, dim=1)


class ModelM5(nn.Module):
    def __init__(self):
        super(ModelM5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5, bias=False)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 5, bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, 5, bias=False)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.fc1 = nn.Linear(10240, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)

    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        flat5 = torch.flatten(conv5.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat5))
        return logits

    def forward(self, x):
        logits = self.get_logits(x)
        return logits
        # return F.log_softmax(logits, dim=1)


class ModelM7(nn.Module):
    def __init__(self):
        super(ModelM7, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, 7, bias=False)    # output becomes 22x22
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 7, bias=False)   # output becomes 16x16
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 7, bias=False)  # output becomes 10x10
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 7, bias=False)  # output becomes 4x4
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(3072, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)

    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits

    def forward(self, x):
        logits = self.get_logits(x)
        return logits
        # return F.log_softmax(logits, dim=1)


# Model for Quantization Aware Training ====================================
class QModel5(nn.Module):
    def __init__(self):
        # super(ModelM5, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5, bias=False)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 5, bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, 5, bias=False)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.fc1 = nn.Linear(10240, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        # x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        # flat5 = torch.flatten(conv5.permute(0, 2, 3, 1), 1)
        flat5 = torch.flatten(conv5, 1)
        logits = self.fc1_bn(self.fc1(flat5))

        x = self.dequant(logits)
        return x
        # return logits


# ANOTHER MODELS
class WideModelM5(nn.Module):
    """
    Takes a lot of time 
    Also have accuracy lower than ordinary ModelM5 after first epoch
    """

    def __init__(self):
        super(WideModelM5, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, bias=False)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, bias=False)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 3, bias=False)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 3, bias=False)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 512, 3, bias=False)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 256, 3, bias=False)
        self.conv6_bn = nn.BatchNorm2d(256)

        # self.fc1 = nn.Linear(10240, 10, bias=False)
        self.fc1 = nn.Linear(65536, 10, bias=False)
        # self.fc1_bn = nn.BatchNorm1d(10)

    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        # conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))

        # flat = torch.flatten(conv6.permute(0, 2, 3, 1), 1)
        flat = torch.flatten(conv6, start_dim=1)
        # logits = self.fc1_bn(self.fc1(flat))
        logits = self.fc1(flat)
        return logits

    def forward(self, x):
        logits = self.get_logits(x)
        return logits
        # return F.log_softmax(logits, dim=1)


class ModelV6(nn.Module):
    def __init__(self) -> None:
        super(ModelV6, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*14*14, out_features=256, bias=False),
            nn.Linear(in_features=256, out_features=10, bias=False),
        )

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        res = self.conv4(res)
        res = self.conv5(res)
        res = self.conv6(res)
        res = self.conv7(res)

        res = torch.flatten(res, start_dim=1)
        res = self.fc(res)
        return res


def ResNet():
    model = resnet50()

    # Change first layer to fit model to Mnist task
    model.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(
        7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model


def GoogleNet():
    model = googlenet()
    return model
