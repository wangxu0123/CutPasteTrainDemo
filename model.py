import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
import cv2


class ProjectionNet(nn.Module):
    def __init__(self, pretrained=True, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=2):
        super(ProjectionNet, self).__init__()
        # self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
        self.resnet18 = resnet18(pretrained=pretrained)

        #     self.gradients = list()
        #
        # def save_gradient(self, grad):
        #     self.gradients.append(grad)
        #
        # def get_gradients(self):
        #     return self.gradients

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        # the last layer without activation

        head = nn.Sequential(
            *sequential_layers
        )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        embeds = self.resnet18(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)

        # 新增函数
        # 新增
        # temp = self.resnet18.layer4(x)
        # weight_softmax = torch.flatten(embeds, 1)
        source_x = x  # 保存下原图
        for name, module in self.resnet18._modules.items():  # 遍历的方式遍历网络的每一层
            x = module(x)  # input 会经过遍历的每一层
            if name in ['layer4']:
                break
        conv_output = x

        # def reneturnCAM(feature_conv, weight_softmax, class_idx):
        #     # generate the class activation maps upsample to 256×256
        #     size_upsample = (256, 256)
        #     bz, nc, h, w = feature_conv.shape
        #     feature_temp = feature_conv[0]
        #     output_cam = []
        #     for idx in class_idx:
        #         # weight_softmax = torch.flatten(embeds, 1)
        #         cam = weight_softmax[idx].dot(feature_temp.reshape((nc, h * w)))
        #         cam = cam.reshape(h, w)
        #         cam = cam - np.min(cam)
        #         cam_img = cam / np.max(cam)
        #         cam_img = np.uint8(255 * cam_img)
        #         output_cam.append(cv2.resize(cam_img, size_upsample))
        #     return output_cam
        #
        # temp_cam = reneturnCAM(conv_output, embeds, 64)

        return embeds, logits, conv_output

    def freeze_resnet(self):
        # freez full resnet18
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
