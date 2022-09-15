import argparse
import numpy as np
import cv2
import time
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='/home/zyp/pythonProject/L2CS-Net-main/models/_epoch_50.pth.tar', type=str)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    batch_size = 1
    cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = getArch(arch, 28)

    snapshot_path1='/home/zyp/pythonProject/L2CS-Net-main/output/snapshots_backup/L2CS-mpiigaze_1661402948/fold1/_epoch_3.pth.tar'
    snapshot_path2 ='/home/zyp/pythonProject/L2CS-Net-main/output/snapshots_backup/L2CS-mpiigaze_1661406859/fold6/_epoch_15.pth.tar'

    model1=copy.deepcopy(model)
    print('Loading snapshot.')
    saved_state_dict1 = torch.load(snapshot_path1)
    model1.load_state_dict(saved_state_dict1)

    m_dic1=model1.state_dict()
    #print(model1.state_dict())
    for k,v in m_dic1.items():
        if k=='layer1.0.conv1.weight':
            print('{}:{}:{}'.format(k,v.shape,v))

    model1=copy.deepcopy(model)
    print('Loading snapshot.')
    saved_state_dict1 = torch.load(snapshot_path2)
    model1.load_state_dict(saved_state_dict1)

    m_dic1=model1.state_dict()
    #print(model1.state_dict())
    for k,v in m_dic1.items():
        if k=='layer1.0.conv1.weight':
            print('{}:{}:{}'.format(k,v.shape,v))
