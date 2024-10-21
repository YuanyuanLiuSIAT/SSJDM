import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import ifft2c, fft2c, r2c, c2r, Emat_xyt
from models.Bayesconv import  BBBConv2d_v2


class BResBlock3(torch.nn.Module):
    def __init__(self, device, prior_mu, prior_rho, conv_size=64, nch=24, filter_size=3,pad=1, bias = False):
        super(BResBlock3, self).__init__()

        self.lam = nn.Parameter(torch.Tensor([.5]))
        self.eta = nn.Parameter(torch.Tensor([.5]))
        self.tau = nn.Parameter(torch.Tensor([.5]))
        self.conv1 = nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv2 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv3 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv4 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv5 = BBBConv2d_v2(device=device,in_channels=conv_size, out_channels=nch, kernel_size=filter_size, padding=1,bias=bias,prior_mu=prior_mu,prior_rho=prior_rho)

        self.conv6 = nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv7 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv8 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv9 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv10 = BBBConv2d_v2(device=device,in_channels=conv_size, out_channels=nch, kernel_size=filter_size, padding=1,bias=bias,prior_mu=prior_mu,prior_rho=prior_rho)
        
         
    def forward(self, y, atb, csm, mask):
        nb, nc, nt, nx, ny = y.size()
        y = y.view(nb, nc*nt, nx, ny)
        x = self.conv1(y)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x, mu1, sigma1 = self.conv5(x)
        lam = torch.clamp(self.lam, 0.01, 0.98)
        x = lam * y + (.99 - lam) * x

        y = r2c(y)
        y = fft2c(y)
        y = c2r(y)
        z = self.conv6(y)
        z = F.relu(z)
        z = self.conv7(z)
        z = F.relu(z)
        z = self.conv8(z)
        z = F.relu(z)
        z = self.conv9(z)
        z = F.relu(z)
        z, mu2, sigma2 = self.conv10(z)

        eta = torch.clamp(self.eta, 0.01, 0.98)
        z = eta * y + (.99 - eta) * z
        z = r2c(z)
        z = ifft2c(z)
        z = c2r(z)

        tau = torch.clamp(self.tau, 0.01, 0.99)
        y = tau * x + (1-tau)*z
        y = y.view(nb,nc,nt,nx,ny)
        y = Emat_xyt(y, False, csm, 1-mask) + atb  #
        y = Emat_xyt(y, True, csm, 1)
        return y,mu1,mu2,sigma1,sigma2

class BPOCS3(torch.nn.Module):
    def __init__(self, config_model):
        super(BPOCS3, self).__init__()
        self.prior_mu = config_model.prior_mu
        self.prior_rho = config_model.prior_rho
        self.device = config_model.device

        self.LayerNo = config_model.LayerNo
        self.nch_in = config_model.nch_in
        self.nch_out = config_model.nch_out
        self.nch_ker = config_model.nch_ker
        self.filter_size = config_model.filter_size
        self.bias = config_model.bias
        self.pad = config_model.pad
        Layers = []

        for i in range(self.LayerNo):
            Layers.append(BResBlock3(device=self.device, prior_mu=self.prior_mu, prior_rho=self.prior_rho, conv_size=self.nch_ker,nch=self.nch_in, filter_size = self.filter_size, pad=self.pad,bias=self.bias))

        self.fcs = nn.ModuleList(Layers)

    def forward(self, atb, csm, mask):
        x = Emat_xyt(atb, True, csm, mask)
        mu = []
        sigma = []
        for i in range(self.LayerNo):
            x, mu1, mu2, sigma1, sigma2 = self.fcs[i](x, atb, csm, mask)
            mu += [mu1]
            mu += [mu2]
            sigma += [sigma1]
            sigma += [sigma2]
        return x,mu,sigma