import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64     #각 block의 입력 채널 수
        super(L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)    #Batch Normlization
        self.relu = nn.ReLU(inplace=True)  #self.conv에서 계산한 결과를 그대로 수정하면 메모리를 절약할 수 있고, 메모리를 반복적으로 요청 및 방출하는 시간을 절약할 수 있으며, 최종 계산 결과는 inplace=False와 같습니다.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #maxpool处理
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  #풀링된 각 채널의 크기는 1x1입니다.  각 채널에는 픽셀 하나만 있습니다.(1，1 )은 outputsize를 나타냅니다.

        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)  #얻은 특징을 분류합니다.

       # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():   #가중치를 초기화합니다
            if isinstance(m, nn.Conv2d):#首先应该声明张量，然后修改张量的权重。通过调用torch.nn.init包中的多种方法可以将权重初始化为直接访问张量的属性。
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):    #stride>1 또는 입출력 채널의 수가 다르면 다운샘플링합니다.
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  #위에 basicblok이 정의되어 있지 않아서 expansion만 가능합니다.
            downsample = nn.Sequential(                               #expandsion은 다음 BasicBlock의 입력 채널이 얼마인지 지정하는 데 사용합니다.
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False), # 하나의 kernel=1의 conv2d 컨볼루션 core를 사용하여 채널 정보를 낮추고 H/W 스케일도 다르면 stride를 설계합니다.
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # 여기서 basicblok이 생겼습니다
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #다차원적인 tensor를 1차원으로 펴줍니다

        
        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze



