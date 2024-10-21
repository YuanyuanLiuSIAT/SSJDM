import torch
import torch.nn as nn

class UNet_YE(nn.Module):
    def __init__(self, config):
        super(UNet_YE, self).__init__()

        self.nch_in = config.model.nch_in
        self.nch_out = config.model.nch_out
        self.nch_ker = config.model.nch_ker
        self.is_pool = config.model.is_pool

        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)
        self.maxpool6 = nn.MaxPool2d(2)
        self.maxpool8 = nn.MaxPool2d(2)
        
        self.enc1 = nConv2d(1 * self.nch_in,  1 * self.nch_ker, relu=0.0, norm=[], pool=[])
        self.enc2 = nConv2d(1 * self.nch_ker, 1 * self.nch_ker, relu=0.0, norm=[], pool=[])
        self.enc3 = nConv2d(1 * self.nch_ker, 2 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.enc4 = nConv2d(2 * self.nch_ker, 2 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.enc5 = nConv2d(2 * self.nch_ker, 4 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.enc6 = nConv2d(4 * self.nch_ker, 4 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.enc7 = nConv2d(4 * self.nch_ker, 8 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.enc8 = nConv2d(8 * self.nch_ker, 8 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.enc9 = nConv2d(8 * self.nch_ker, 16 * self.nch_ker, relu=0.0, norm=0, pool=[])

        self.dec9 = nConv2d(2 * 8 * self.nch_ker, 2*8 * self.nch_ker, relu=0.0, norm=0, pool=0,out_pool=8 * self.nch_ker)
        self.dec8 = nConv2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.dec7 = nConv2d(1 * 8 * self.nch_ker, 8 * self.nch_ker, relu=0.0, norm=0, pool=0,out_pool=4 * self.nch_ker)
        self.dec6 = nConv2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.dec5 = nConv2d(1 * 4 * self.nch_ker, 4 * self.nch_ker, relu=0.0, norm=0, pool=0,out_pool=2 * self.nch_ker)
        self.dec4 = nConv2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.dec3 = nConv2d(1 * 2 * self.nch_ker, 2 * self.nch_ker, relu=0.0, norm=0, pool=0,out_pool=1 * self.nch_ker)
        self.dec2 = nConv2d(1 * 2 * self.nch_ker, 1 * self.nch_ker, relu=0.0, norm=0, pool=[])
        self.dec1 = nConv2d(1 * 1 * self.nch_ker, 1 * self.nch_ker, relu=0.0, norm=0, pool=[])
        
        self.dec0 = nn.Conv2d(self.nch_ker, self.nch_out, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        down2 = self.maxpool2(enc2)
        enc3 = self.enc3(down2)
        enc4 = self.enc4(enc3)
        down4 = self.maxpool4(enc4)
        enc5 = self.enc5(down4)
        enc6 = self.enc6(enc5)
        down6 = self.maxpool6(enc6)
        enc7 = self.enc7(down6)
        enc8 = self.enc8(enc7)
        down8 = self.maxpool8(enc8)
        enc9 = self.enc9(down8)
       
  

        dec9 = self.dec9(enc9)
        if self.is_pool:
            dec9 = nn.ZeroPad2d(padding=(0,0,0,1))(dec9)
     
        dec8 = self.dec8(torch.cat([enc8, dec9], dim=1))
        dec7 = self.dec7(dec8)
        #if self.is_pool:
        #    dec7 = nn.ZeroPad2d(padding=(1,0,0,0))(dec7)
        dec6 = self.dec6(torch.cat([enc6, dec7], dim=1))
        dec5 = self.dec5(dec6)
        #if self.is_pool:
        #    dec5 = nn.ZeroPad2d(padding=(1,0,0,0))(dec5)
        dec4 = self.dec4(torch.cat([enc4, dec5], dim=1))
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(dec2)

        x = self.dec0(dec1)
        return x


class nConv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=3, stride=1, padding=1, bias=False, relu=0.0, norm=0, pool=1, out_pool=0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm != []:
            layers += [nn.InstanceNorm2d(nch_out)]

        if relu != []:
            layers += [nn.LeakyReLU(0.2,True)]

        if pool != []:
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            layers += [nn.Conv2d(nch_out, out_pool, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)