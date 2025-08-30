import torch
import torch.nn as nn
import torch.nn.functional as F

class Basicconv2d(nn.Module):
    def __init__(self,in_channel,out_channel,**kwargs):
        super(Basicconv2d,self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,**kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Inception(nn.Module):
    def __init__(self,in_channel,ch11,ch33red,ch33,ch55red,ch55,pool_proj):
        super(Inception,self).__init__()
        self.branch1 = Basicconv2d(in_channel,ch11,1)
        self.branch2 = nn.Sequential(
            Basicconv2d(in_channel,ch33red,1),
            Basicconv2d(ch33red,ch33,3,padding=1)
        )
        self.branch3 = nn.Sequential(
            Basicconv2d(in_channel,ch55red,1),
            Basicconv2d(ch55red,ch55,5,padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3,stride=1,padding=1),
            Basicconv2d(in_channel,pool_proj,1)
        )
    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        out = [x1,x2,x3,x4]
        return torch.cat(x,1)

class InceptionAux(nn.Module):
    def __init__(self,in_channel,num_classes):
        super(InceptionAux,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(kernel_size=5,stride=3)
        self.conv = nn.Conv2d(in_channel,128,1)
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,num_classes)
    def forward(self,x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = F.dropout(x,0.5,training=self.training)
        x = F.relu(self.fc1(x),inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)

        return x

class GoogLeNet(nn.Module):
    def __init__(self,num_classes=1000,aux_logits=True):
        super(GoogLeNet,self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.maxpool1 = nn.MaxPool2d(3,2,ceil_mode=True)
        self.conv2 = nn.Conv2d(64,64,1)
        self.conv3 = nn.Conv2d(64,192,3,padding=1)

        self.maxpool2 = nn.MaxPool2d(3,stride=2,ceil_mode=True)
        self.inception3a = Inception(192,64,96,128,16,32,32)
        self.inception3b = Inception(256,128,128,192,32,96,64)
        self.maxpool3 = nn.MaxPool2d(3,2,ceil_mode=True)

        self.inception4a = Inception(480,192,96,208,16,48,64)
        self.inception4b = Inception(512,160,112,224,24,64,64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3,2,ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512,num_classes)
            self.aux2 = InceptionAux(528,num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024,num_classes)


    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.aux_logits:
            return x,aux2,aux1
        return x

