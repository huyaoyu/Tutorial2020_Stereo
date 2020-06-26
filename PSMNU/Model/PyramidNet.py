
from __future__ import print_function

# Some structures of this file is inspired by 
# the research work done by Chang and Chen, 
# Pyramid Stereo Matching Network, CVPR2018.
# https://github.com/JiaRenChang/PSMNet
# 

import cv2
import math
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PyramidNetException(Exception):
    def __init__(self, msg):
        super(PyramidNetException, self).__init__(msg)

class Inspector(object):
    def __init__(self, workingDir):
        self.workingDir = workingDir

    def initialize_working_dir(self, wd=None):
        if ( wd is not None ):
            self.workingDir = wd
        
        if ( False == os.path.isdir(self.workingDir) ):
            os.makedirs(self.workingDir)

    def save_real_value_image(self, fn, img):
        """
        img will be normalized and scaled to 0-255. Then a
        single channel image will be created and then witten 
        to the filesystem.

        img must be a NumPy array.

        fn is a the filename without the extension.
        """

        img = ( img - img.min() ) / ( img.max() - img.min() ) * 255
        img = img.astype( np.uint8 )

        # Full fill the filename.
        fn = "%s.png" % (fn)

        # Save the image by OpenCV.
        cv2.imwrite(fn, img)

    def save_tensor_as_images(self, t, prefix):
        """
        t: The tensor. It is assumed that t is either a 4D or 5D tensor.
        """
        
        dims = len( t.size() )

        if ( 4 != dims and 5 != dims ):
            raise Exception("t must be a 4D or 5D tensor. dims = %d." % (dims))

        # Find out the dimensions.
        if ( 4 == dims ):
            D = 1
        else:
            D = t.size()[2]

        N = t.size()[0]
        C = t.size()[1]
        H = t.size()[-2]
        W = t.size()[-1]

        # Transfer data to CPU.
        tc = t.cpu()

        # View the tensor as a 5D tensor.
        tc = tc.view( N, C, D, H, W )

        for n in range(N):
            for c in range(C):
                for d in range(D):
                    # Convert a single plane into a NumPy array.
                    img = tc[n,c,d,:,:].numpy()

                    # Compose the filename.
                    fn = "%s/%s_%d.%d.%d" % ( self.workingDir, prefix, n, c, d )

                    # Save the plane as an image.
                    self.save_real_value_image(fn, img)

class ConvBN(nn.Module):
    def __init__(self, inChannels, outChannels, kSize, stride, padding, dilation):
        super( ConvBN, self ).__init__()

        self.inChannels  = inChannels
        self.outChannels = outChannels
        self.kSize       = kSize
        self.stride      = stride
        self.padding     = padding
        self.dilation    = dilation

    def forward(self, x):
        raise PyramidNetException("Base class abstract member function is called.")
    
class CB2D(ConvBN):
    def __init__(self, inChannels, outChannels, kSize, stride, padding, dilation):
        super(CB2D, self).__init__( inChannels, outChannels, kSize, stride, padding, dilation )

        # Modify the values of padding.
        self.padding = self.dilation if self.dilation > 1 else self.padding

        # The nn.
        self.cb = nn.Sequential( \
            nn.Conv2d( self.inChannels, self.outChannels, kernel_size=self.kSize, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=False ), \
            nn.BatchNorm2d( self.outChannels ) )
        
    def forward(self, x):
        return self.cb(x)
    
class CB3D(ConvBN):
    def __init__(self, inChannels, outChannels, kSize, stride, padding):
        super(CB3D, self).__init__( inChannels, outChannels, kSize, stride, padding, dilation=1 )

        # Modify the values of padding.
        self.padding = self.dilation if self.dilation > 1 else self.padding

        # The nn.
        self.cb = nn.Sequential( \
            nn.Conv3d( self.inChannels, self.outChannels, kernel_size=self.kSize, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=False ), \
            nn.BatchNorm3d( self.outChannels ) )
        
    def forward(self, x):
        return self.cb(x)

class LayerBlock(nn.Module):
    """
    A residual block. He, et. al., Deep Residual Learning for Image Recognition. CVPR2016.
    """
    def __init__(self, inChannels, outChannels, stride, padding, dilation, featureTransformer=None):
        super(LayerBlock, self).__init__()

        self.kSize      = 3
        self.lastStride = 1

        # The nn.
        self.conv1 = nn.Sequential( \
            CB2D( inChannels, outChannels, self.kSize, stride, padding, dilation ), \
            nn.ReLU( inplace=True ) )

        self.conv2 = CB2D( outChannels, outChannels, self.kSize, self.lastStride, padding, dilation )

        self.featureTransformer = featureTransformer

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if ( self.featureTransformer is not None ):
            x = self.featureTransformer(x)

        out += x

        return out

class FeatureExtractionLayer(nn.Module):
    def __init__(self, block, inChannels, outChannels, nBlocks, stride, padding, dilation):
        super(FeatureExtractionLayer, self).__init__()

        ft = None # The feature transformer.
        if ( 1 != stride or inChannels != outChannels ):
            # An actual feature transformer is needed here.
            # Check the output dimension of ft.
            ft = CB2D( inChannels, outChannels, kSize=1, stride=stride, padding=0, dilation=1 )
        
        layers = []
        layers.append( block( inChannels, outChannels, stride, padding, dilation, featureTransformer=ft ) )

        for i in range(nBlocks-1):
            layers.append( block( outChannels, outChannels, 1, padding, dilation, featureTransformer=None ) )

        self.fel = nn.Sequential( *layers )

    def forward(self, x):
        return self.fel(x)

class SPPBranch(nn.Module):
    def __init__(self, inChannels, outChannels, avgPoolingKSize):
        super(SPPBranch, self).__init__()

        self.inChannels  = inChannels
        self.outChannels = outChannels
        self.APSK        = avgPoolingKSize

        # The nn.
        self.branch = nn.Sequential( \
            nn.AvgPool2d( (self.APSK, self.APSK), stride=(self.APSK, self.APSK) ), \
            CB2D( self.inChannels, self.outChannels, 1, 1, 0, 1 ), \
            nn.ReLU( inplace=True ) )
        
    def forward(self, x):
        return self.branch(x)

class FeatureExtraction(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(FeatureExtraction, self).__init__()

        self.inChannels  = inChannels
        self.outChannels = outChannels

        self.postConvIntermediateChannels = 128

        # The nns.
        # Pre-process convolutions.
        self.preConv = nn.Sequential( \
            CB2D( self.inChannels, self.outChannels, 3, 2, 1, 1 ), \
            nn.ReLU( inplace=True ), \
            CB2D( self.outChannels, self.outChannels, 3, 1, 1, 1 ), \
            nn.ReLU( inplace=True ), \
            CB2D( self.outChannels, self.outChannels, 3, 1, 1, 1 ), \
            nn.ReLU( inplace=True ) )
        
        # Feature extraction layers.
        self.fe1 = FeatureExtractionLayer( LayerBlock,   self.outChannels,   self.outChannels,  3, 1, 1, 1 )
        self.fe2 = FeatureExtractionLayer( LayerBlock,   self.outChannels, 2*self.outChannels, 16, 2, 1, 1 )
        self.fe3 = FeatureExtractionLayer( LayerBlock, 2*self.outChannels, 4*self.outChannels,  3, 1, 1, 1 )
        self.fe4 = FeatureExtractionLayer( LayerBlock, 4*self.outChannels, 4*self.outChannels,  3, 1, 1, 2 )

        # SPP.
        self.SPP1 = SPPBranch( 4*self.outChannels, self.outChannels, 64 )
        self.SPP2 = SPPBranch( 4*self.outChannels, self.outChannels, 32 )
        self.SPP3 = SPPBranch( 4*self.outChannels, self.outChannels, 16 )
        self.SPP4 = SPPBranch( 4*self.outChannels, self.outChannels,  8 )

        # Post-process convolutions.
        self.postConv = nn.Sequential( \
            CB2D( 2*self.outChannels + 4*self.outChannels + self.outChannels * 4, self.postConvIntermediateChannels, 3, 1, 1, 1 ), \
            nn.ReLU( inplace=True ),
            nn.Conv2d( self.postConvIntermediateChannels, self.outChannels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False ) )

    def forward(self, x):
        # Pre-process convolutions.
        output = self.preConv(x)

        # Feature extraction.
        output     = self.fe1(output)
        outputRaw  = self.fe2(output)
        output     = self.fe3(outputRaw)
        outputSkip = self.fe4(output)

        # SPP.
        outputSPP1 = self.SPP1(outputSkip)
        outputSPP1 = F.interpolate( outputSPP1, ( outputSkip.size()[2], outputSkip.size()[3] ), mode="bilinear", align_corners=False )
        outputSPP2 = self.SPP1(outputSkip)
        outputSPP2 = F.interpolate( outputSPP2, ( outputSkip.size()[2], outputSkip.size()[3] ), mode="bilinear", align_corners=False )
        outputSPP3 = self.SPP1(outputSkip)
        outputSPP3 = F.interpolate( outputSPP3, ( outputSkip.size()[2], outputSkip.size()[3] ), mode="bilinear", align_corners=False )
        outputSPP4 = self.SPP1(outputSkip)
        outputSPP4 = F.interpolate( outputSPP4, ( outputSkip.size()[2], outputSkip.size()[3] ), mode="bilinear", align_corners=False )

        # Concat the extracted features.
        output = torch.cat( ( outputRaw, outputSkip, outputSPP4, outputSPP3, outputSPP2, outputSPP1 ), 1 )

        # Post-process convolutions.
        output = self.postConv( output )

        return output

class Hourglass(nn.Module):
    def __init__(self, inChannels):
        super(Hourglass, self).__init__()

        self.inChannels = inChannels

        # The nns.
        self.conv1 = nn.Sequential( \
            CB3D( self.inChannels, 2*self.inChannels, 3, 2, 1 ), \
            nn.ReLU( inplace=True ) )
        
        self.conv2 = CB3D( 2*self.inChannels, 2*self.inChannels, 3, 1, 1 )

        self.conv3 = nn.Sequential( \
            CB3D( 2*self.inChannels, 2*self.inChannels, 3, 2, 1 ), \
            nn.ReLU( inplace=True ) )

        self.conv4 = nn.Sequential( \
            CB3D( 2*self.inChannels, 2*self.inChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ) )

        self.conv5 = nn.Sequential( \
            nn.ConvTranspose3d( 2*self.inChannels, 2*self.inChannels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False ), \
            nn.BatchNorm3d( 2*self.inChannels ) )

        self.conv6 = nn.Sequential( \
            nn.ConvTranspose3d( 2*self.inChannels, self.inChannels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False ), \
            nn.BatchNorm3d( self.inChannels ) )

    def forward(self, x, preSqu, postSqu, prefix="", inspector=None):
        out = self.conv1(x)
        pre = self.conv2(out)

        if ( postSqu is not None ):
            pre = F.relu( pre + postSqu, inplace=True )
        else:
            pre = F.relu( pre, inplace=True )
        
        out = self.conv3(pre)
        out = self.conv4(out)

        if ( inspector is not None ):
            # Perform inspection.
            inspector.save_tensor_as_images( out, prefix + "c4" )

        if ( preSqu is not None ):
            post = F.relu( self.conv5(out) + preSqu, inplace=True )
        else:
            post = F.relu( self.conv5(out) + pre, inplace=True )
        
        out = self.conv6(post)

        return out, pre, post

class DisparityRegression(nn.Module):
    def __init__(self, maxDisp, flagCPU=False):
        super(DisparityRegression, self).__init__()
        
        if ( not flagCPU ):
            self.disp = Variable( \
                torch.Tensor( \
                    np.reshape( np.array( range(maxDisp) ), [1, maxDisp, 1, 1] ) \
                            ).cuda(),\
                    requires_grad=False )
        else:
            self.disp = Variable( \
            torch.Tensor( \
                np.reshape( np.array( range(maxDisp) ), [1, maxDisp, 1, 1] ) \
                        ),\
                requires_grad=False )

    def forward(self, x):
        disp = self.disp.repeat( x.size()[0], 1, x.size()[2], x.size()[3] )
        out  = torch.sum( x * disp, 1 )
        
        return out

class PSMNet(nn.Module):
    def __init__(self, inChannels, featureChannels, maxDisp):
        super(PSMNet, self).__init__()

        self.maxDisp         = maxDisp
        self.inChannels      = inChannels
        self.featureChannels = featureChannels
        self.flagCPU         = False

        # Feature extraction.
        self.featureExtraction = FeatureExtraction( self.inChannels, self.featureChannels )

        # Prepare for hourglass layers.
        self.ph1 = nn.Sequential( \
            CB3D( 2*self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ), \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ) )
        
        self.ph2 = nn.Sequential( \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ), \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ) )
        
        # Hourglass layers.
        self.hg1 = Hourglass( self.featureChannels )
        self.hg2 = Hourglass( self.featureChannels )
        self.hg3 = Hourglass( self.featureChannels )

        # Classification in disparity dimension.
        self.cd1 = nn.Sequential( \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ),
            nn.Conv3d( self.featureChannels, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False ) )

        self.cd2 = nn.Sequential( \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ),
            nn.Conv3d( self.featureChannels, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False ) )

        self.cd3 = nn.Sequential( \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ),
            nn.Conv3d( self.featureChannels, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False ) )

        # Weights initialization.
        for m in self.modules():
            if ( isinstance( m, (nn.Conv2d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt( 2.0 / n ))
            elif ( isinstance( m, (nn.Conv3d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt( 2.0 / n ))
            elif ( isinstance( m, (nn.BatchNorm2d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.BatchNorm3d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.Linear) ) ):
                m.bias.data.zero_()
            # else:
            #     raise PyramidNetException("Unexpected module type {}.".format(type(m)))

    def set_cpu_mode(self):
        self.flagCPU = True

    def unset_cpu_mode(self):
        self.flagCPU = False

    def forward(self, L, R):
        # Feature extraction.

        refFeature = self.featureExtraction(L)
        tgtFeature = self.featureExtraction(R)

        # Make new cost volume as 5D tensor.
        cost = Variable( \
            torch.FloatTensor( refFeature.size()[0], refFeature.size()[1]*2, int(self.maxDisp/4), refFeature.size()[2], refFeature.size()[3]).zero_() \
                       )

        if ( not self.flagCPU ):               
            cost = cost.cuda()

        for i in range( int(self.maxDisp / 4) ):
            if i > 0 :
                cost[:, :refFeature.size()[1],  i, :, i:] = refFeature[ :, :, :, i:   ]
                cost[:,  refFeature.size()[1]:, i, :, i:] = tgtFeature[ :, :, :,  :-i ]
            else:
                cost[:, :refFeature.size()[1],  i, :, :] = refFeature
                cost[:,  refFeature.size()[1]:, i, :, :] = tgtFeature

        cost = cost.contiguous()

        # Prepare for hourglass layers.
        cost0 = self.ph1(cost)
        cost0 = self.ph2(cost0) + cost0

        # Hourglass layers.
        out1, pre1, post1 = self.hg1( cost0, None, None )
        # import ipdb; ipdb.set_trace()
        out1 = out1 + cost0

        out2, pre2, post2 = self.hg2( out1, pre1, post1 )
        out2 = out2 + cost0

        out3, pre3, post3 = self.hg3( out2, pre1, post2 )
        out3 = out3 + cost0

        # Classification in the disparity dimension.
        cost1 = self.cd1( out1 )
        cost2 = self.cd2( out2 ) + cost1
        cost3 = self.cd3( out3 ) + cost2

        # Disparity regression.
        if ( self.training ):
            cost1 = F.interpolate( cost1, [ self.maxDisp, L.size()[2], L.size()[3] ], mode="trilinear", align_corners=False )
            cost1 = torch.squeeze( cost1, 1 )
            pred1 = F.softmax( cost1, dim = 1 )
            pred1 = DisparityRegression( self.maxDisp, self.flagCPU )( pred1 )

            cost2 = F.interpolate( cost2, [ self.maxDisp, L.size()[2], L.size()[3] ], mode="trilinear", align_corners=False )
            cost2 = torch.squeeze( cost2, 1 )
            pred2 = F.softmax( cost2, dim = 1 )
            pred2 = DisparityRegression( self.maxDisp, self.flagCPU )( pred2 )

        cost3 = F.interpolate( cost3, [ self.maxDisp, L.size()[2], L.size()[3] ], mode="trilinear", align_corners=False )
        cost3 = torch.squeeze( cost3, 1 )
        pred3 = F.softmax( cost3, dim = 1 )
        pred3 = DisparityRegression( self.maxDisp, self.flagCPU )( pred3 )

        if ( self.training ):
            return pred1, pred2, pred3
        else:
            return pred3
    
class PSMNetWithUncertainty(nn.Module):
    def __init__(self, inChannels, featureChannels, maxDisp):
        super(PSMNetWithUncertainty, self).__init__()

        self.maxDisp         = maxDisp
        self.inChannels      = inChannels
        self.featureChannels = featureChannels
        self.flagCPU         = False

        # Feature extraction.
        self.featureExtraction = FeatureExtraction( self.inChannels, self.featureChannels )

        # Prepare for hourglass layers.
        self.ph1 = nn.Sequential( \
            CB3D( 2*self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ), \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ) )
        
        self.ph2 = nn.Sequential( \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ), \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ) )
        
        # Hourglass layers.
        self.hg1 = Hourglass( self.featureChannels )
        self.hg2 = Hourglass( self.featureChannels )
        self.hg3 = Hourglass( self.featureChannels )

        # Classification in disparity dimension.
        self.cd1 = nn.Sequential( \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ),
            nn.Conv3d( self.featureChannels, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False ) )

        self.cd2 = nn.Sequential( \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ),
            nn.Conv3d( self.featureChannels, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False ) )

        # ===================================================================
        # ==================== Different with PSMNet. =======================
        # ===================================================================
        self.cd3 = nn.Sequential( \
            CB3D( self.featureChannels, self.featureChannels, 3, 1, 1 ), \
            nn.ReLU( inplace=True ),
            nn.Conv3d( self.featureChannels, 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False ) )

        # Weights initialization.
        for m in self.modules():
            if ( isinstance( m, (nn.Conv2d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt( 2.0 / n ))
            elif ( isinstance( m, (nn.Conv3d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt( 2.0 / n ))
            elif ( isinstance( m, (nn.BatchNorm2d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.BatchNorm3d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.Linear) ) ):
                m.bias.data.zero_()
            # else:
            #     raise PyramidNetException("Unexpected module type {}.".format(type(m)))

    def set_cpu_mode(self):
        self.flagCPU = True

    def unset_cpu_mode(self):
        self.flagCPU = False

    def forward(self, L, R):
        # Feature extraction.

        refFeature = self.featureExtraction(L)
        tgtFeature = self.featureExtraction(R)

        # Make new cost volume as 5D tensor.
        cost = Variable( \
            torch.FloatTensor( refFeature.size()[0], refFeature.size()[1]*2, int(self.maxDisp/4), refFeature.size()[2], refFeature.size()[3]).zero_() \
                       )
        
        if ( not self.flagCPU ):
            cost = cost.cuda()

        for i in range( int(self.maxDisp / 4) ):
            if i > 0 :
                cost[:, :refFeature.size()[1],  i, :, i:] = refFeature[ :, :, :, i:   ]
                cost[:,  refFeature.size()[1]:, i, :, i:] = tgtFeature[ :, :, :,  :-i ]
            else:
                cost[:, :refFeature.size()[1],  i, :, :] = refFeature
                cost[:,  refFeature.size()[1]:, i, :, :] = tgtFeature

        cost = cost.contiguous()

        # Prepare for hourglass layers.
        cost0 = self.ph1(cost)
        cost0 = self.ph2(cost0) + cost0

        # Hourglass layers.
        out1, pre1, post1 = self.hg1( cost0, None, None )
        # import ipdb; ipdb.set_trace()
        out1 = out1 + cost0

        out2, pre2, post2 = self.hg2( out1, pre1, post1 )
        out2 = out2 + cost0

        out3, pre3, post3 = self.hg3( out2, pre1, post2 )
        out3 = out3 + cost0

        # Classification in the disparity dimension.
        cost1 = self.cd1( out1 )
        cost2 = self.cd2( out2 ) + cost1

        # ===================================================================
        # ==================== Different with PSMNet. =======================
        # ===================================================================
        # cost3 = self.cd3( out3 ) + cost2
        lastClassification = self.cd3( out3 )
        cost3 = lastClassification[:, 0, :, :, :].unsqueeze(1) + cost2
        
        logSigmaSquredOverD = lastClassification[:, 1, :, :, :]
        logSigmaSquredOverD = F.interpolate( logSigmaSquredOverD, [ L.size()[2], L.size()[3] ], mode="bilinear", align_corners=False )
        logSigmaSquredOverD = torch.squeeze( logSigmaSquredOverD, 1 )
        logSigmaSquredOverD = torch.mean( logSigmaSquredOverD, 1, keepdim=False )

        # Disparity regression.
        if ( self.training ):
            cost1 = F.interpolate( cost1, [ self.maxDisp, L.size()[2], L.size()[3] ], mode="trilinear", align_corners=False )
            cost1 = torch.squeeze( cost1, 1 )
            pred1 = F.softmax( cost1, dim = 1 )
            pred1 = DisparityRegression( self.maxDisp, self.flagCPU )( pred1 )

            cost2 = F.interpolate( cost2, [ self.maxDisp, L.size()[2], L.size()[3] ], mode="trilinear", align_corners=False )
            cost2 = torch.squeeze( cost2, 1 )
            pred2 = F.softmax( cost2, dim = 1 )
            pred2 = DisparityRegression( self.maxDisp, self.flagCPU )( pred2 )

        cost3 = F.interpolate( cost3, [ self.maxDisp, L.size()[2], L.size()[3] ], mode="trilinear", align_corners=False )
        cost3 = torch.squeeze( cost3, 1 )
        pred3 = F.softmax( cost3, dim = 1 )
        pred3 = DisparityRegression( self.maxDisp, self.flagCPU )( pred3 )

        if ( self.training ):
            return pred1, pred2, pred3, logSigmaSquredOverD
        else:
            return pred3, logSigmaSquredOverD

