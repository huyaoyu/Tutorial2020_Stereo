from __future__ import print_function

import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def is_odd(x):
    r = x%2

    if ( r != 1 ):
        return False
    else:
        return True

def selected_relu(x):
    return F.selu(x, inplace=True)

class SelectedReLU(nn.Module):
    def __init__(self):
        super(SelectedReLU, self).__init__()

        self.model = nn.SELU(True)

    def forward(self, x):
        return self.model(x)

class Conv_W(nn.Module):
    def __init__(self, inCh, outCh, k=3, activation=None):
        """
        k must be an odd number.
        """
        super(Conv_W, self).__init__()

        moduleList = [ \
            nn.ReflectionPad2d(padding=k//2), 
            nn.Conv2d(inCh, outCh, kernel_size=k, stride=1, padding=0, dilation=1, bias=True)
        ]

        if ( activation is not None ):
            moduleList.append( activation )

        self.model = nn.Sequential( *moduleList )

    def forward(self, x):
        return self.model(x)

class Conv_Half(nn.Module):
    def __init__(self, inCh, outCh, k=3, activation=None):
        """
        k must be an odd number.
        """
        super(Conv_Half, self).__init__()

        moduleList = [ \
            nn.ReflectionPad2d(padding=k//2), 
            nn.Conv2d(inCh, outCh, kernel_size=k, stride=2, padding=0, dilation=1, bias=True)
        ]

        if ( activation is not None ):
            moduleList.append( activation )

        self.model = nn.Sequential( *moduleList )

    def forward(self, x):
        return self.model(x)

class Deconv_DoubleSize(nn.Module):
    def __init__(self, inCh, outCh, k=3, p=1, op=0, activation=None):
        super(Deconv_DoubleSize, self).__init__()

        moduleList = [ nn.ConvTranspose2d(inCh, outCh, kernel_size=k, stride=2, padding=p, dilation=1, output_padding=op, bias=True) ]

        if ( activation is not None ):
            moduleList.append( activation )

        self.model = nn.Sequential( *moduleList )

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, inCh, k, scale=None, activation=None, lastActivation=None):
        super(ResBlock, self).__init__()

        if ( activation is not None ):
            self.model = nn.Sequential( \
                Conv_W(inCh, inCh, k), \
                activation, \
                Conv_W(inCh, inCh, k) )
        else:
            self.model = nn.Sequential( \
                Conv_W(inCh, inCh, k), \
                Conv_W(inCh, inCh, k) )
        
        self.scale = scale
        self.lastActivation = lastActivation

    def forward(self, x):
        if ( self.scale is not None ):
            res = self.model(x).mul( self.scale )
        else:
            res = self.model(x)
        
        res += x

        if ( self.lastActivation is not None ):
            return self.lastActivation(res)
        else:
            return res

class ResPack(nn.Module):
    def __init__(self, inCh, outCh, n, k, scale=None, bypass=False, activation=None, lastActivation=None):
        super(ResPack, self).__init__()

        m = []

        for i in range(n):
            m.append( ResBlock( inCh, k, scale, activation=activation, lastActivation=activation ) )

        if ( bypass and inCh != outCh ):
            raise Exception("When using bypass, inCh({}) must equals outCh({}). ".format( inCh, outCh ))
            
        m.append( Conv_W( inCh, outCh, k ) )

        self.model = nn.Sequential(*m)
        self.bypass = bypass
        self.lastActivation = lastActivation

    def forward(self, x):
        res = self.model(x)
        
        if ( self.bypass ):
            res += x

        if ( self.lastActivation is not None ):
            return self.lastActivation(res)
        else:
            return res

class ReceptiveBranch(nn.Module):
    def __init__(self, inCh, outCh, r, activation=None):
        super(ReceptiveBranch, self).__init__()

        moduleList = [ \
            nn.AvgPool2d( kernel_size=r, stride=r, padding=0, ceil_mode=False, count_include_pad=True ), \
            nn.Conv2d( inCh, outCh, kernel_size=1, stride=1, padding=0, dilation=1, bias=True ) \
        ]

        if ( activation is not None ):
            moduleList.append( activation )

        self.model = nn.Sequential( *moduleList )

    def forward(self, x):
        return self.model(x)

class ConvBranch(nn.Module):
    def __init__(self, inCh, outCh, n, r, activation=None):
        super(ConvBranch, self).__init__()

        if ( n <= 0 ):
            raise Exception("n = {}. ".format(n))

        moduleList = []

        for i in range(n):
            moduleList.append( Conv_W(inCh, inCh, k=3, activation=activation) )
            moduleList.append( nn.AvgPool2d( kernel_size=r, stride=r, padding=0, ceil_mode=False, count_include_pad=True ) )
        
        moduleList.append( Conv_W(inCh, outCh, k=3, activation=activation) )

        self.model = nn.Sequential( *moduleList )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    print("Test CommonModel.py")

    inCh  = 32
    outCh = 64
    k     = 3

    conv_w    = Conv_W(inCh, outCh, k)
    resBlock  = ResBlock( inCh, outCh, k )
    resPack   = ResPack( inCh, outCh, 2, k )
    recBranch = ReceptiveBranch( inCh, outCh, 4 )
