import sys
sys.path.append("..")

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from models.misc import ModuleWrapper


class BBBConv2d_v2(ModuleWrapper):
    def __init__(self, device, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, prior_mu=(0,1), prior_rho=(0,0)):

        super(BBBConv2d_v2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = device
        self.posterior_mu_initial = prior_mu
        self.posterior_rho_initial = prior_rho

        self.W_mu = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_rho = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(self.posterior_mu_initial[0],self.posterior_mu_initial[1])#
        #self.W_rho.data.normal_(*self.posterior_rho_initial)
        self.W_rho.data.uniform_(self.posterior_rho_initial[0],self.posterior_rho_initial[1])

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            #self.bias_rho.data.normal_(*self.posterior_rho_initial)
            self.bias_rho.data.uniform_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias = self.bias_mu
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups),self.W_mu,self.W_sigma

