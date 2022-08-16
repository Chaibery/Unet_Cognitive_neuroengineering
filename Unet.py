import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):#Unet结构
    def __init__(self,channels,classes):
        super(Unet, self).__init__()
        self.n_channels = channels
        self.n_classes = classes

        self.start = doubleconv(self.n_channels, 64)
        self.pool1 = max_pool(64, 128)
        self.pool2 = max_pool(128, 256)
        self.pool3 = max_pool(256, 512)
        self.pool4 = max_pool(512, 1024)
        self.up1 = up_conv(1024, 512)
        self.up2 = up_conv(512, 256)
        self.up3 = up_conv(256, 128)
        self.up4 = up_conv(128, 64)
        self.out = OutPut_SEG(64, self.n_classes)

    def forward(self, x):
        x1 = self.start(x)
        aftpool1 = self.pool1(x1)
        aftpool2 = self.pool2(aftpool1)
        aftpool3 = self.pool3(aftpool2)
        aftpool4 = self.pool4(aftpool3)
        x = self.up1(aftpool4, aftpool3)
        x = self.up2(x, aftpool2)
        x = self.up3(x, aftpool1)
        x = self.up4(x, x1)
        outcome = self.out(x)
        return outcome

class OutPut_SEG(nn.Module):#最后的单层输出层
    def __init__(self, in_channels, out_channels):
        super(OutPut_SEG, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class doubleconv(nn.Module):#连续两次卷积
    def __init__(self,input,output):
        super(doubleconv,self).__init__()
        self.double_conv = nn.Sequential(
        nn.Conv2d(input, output, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU(inplace=True),
        nn.Conv2d(output, output, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class max_pool(nn.Module):#下采样2*2
    def __init__(self, input, output):
        super(max_pool,self).__init__()
        self.max_pooling = nn.Sequential(
            nn.MaxPool2d(2),
            doubleconv(input, output)
        )

    def forward(self, x):
        return self.max_pooling(x)

class up_conv(nn.Module):#进行上采样2*2
    def __init__(self, input, output):
        super(up_conv,self).__init__()
        self.up = nn.ConvTranspose2d(input, input // 2, kernel_size=2, stride=2)
        self.conv = doubleconv(input, output)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        DY = x2.size()[2] - x1.size()[2]
        DX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [DX // 2, DX - DX // 2,DY // 2, DY - DY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)








