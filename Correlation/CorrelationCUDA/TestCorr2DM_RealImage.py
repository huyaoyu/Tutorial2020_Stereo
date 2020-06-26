from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch

import Corr2D
import Corr2D_ext

def test_gradcheck(B=2, C=2, H=4, W=4, \
        padding=1, kernelSize=3, maxDisplacement=1, strideK=1, strideD=1):

    print("test_backward()")

    # Random tensor.
    t0 = torch.ones((B, C, H, W)).double().cuda()
    t1 = t0.clone().detach()

    t0.requires_grad = True
    t1.requires_grad = True

    # Create a Corr2DM object.
    corr2d = Corr2D.Corr2DM( maxDisplacement, padding=padding, kernelSize=kernelSize, strideK=strideK, strideD=strideD )

    # Check gradient.
    test = torch.autograd.gradcheck( corr2d, ( t0, t1 ), eps=1e-3, atol=1e-6 )
    print(test)

def test_two_tensors(t0, t1, \
    padding=1, kernelSize=3, maxDisplacement=64, strideK=1, strideD=1):
    
    # Create a Corr2DM object.
    corr2d = Corr2D.Corr2DM( maxDisplacement, padding=padding, kernelSize=kernelSize, strideK=strideK, strideD=strideD )

    out = corr2d( t0, t1 )

    return out

def zero_normalized_cross_correlation(x0, x1):
    x0 = x0.reshape((1,-1))
    x1 = x1.reshape((-1,1))

    n0 = np.linalg.norm(x0)
    n1 = np.linalg.norm(x1)

    nx0 = x0 / n0
    nx1 = x1 / n1

    return nx0.dot(nx1)

def test_real_image_shift(shift=0):
    # img0 = cv2.imread("BoxN.png", cv2.IMREAD_UNCHANGED)
    # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    
    img0 = np.zeros((256, 256), dtype=np.float64) + 1.0
    img0[ 128:134, 128:134 ] = -1.0

    h = img0.shape[0]
    w = img0.shape[1]

    img0 = img0.reshape((h, w, 1))

    # img0 = img0.reshape((h, w, 1))

    img1 = np.zeros_like(img0)

    # Shift img0.
    img1[:, :w-shift] = img0[:, shift:]

    # Convert the images into tensors.
    t0 = torch.from_numpy(img0).double().permute((2, 0, 1)).unsqueeze(0).cuda()
    t1 = torch.from_numpy(img1).double().permute((2, 0, 1)).unsqueeze(0).cuda()

    t0.requires_grad = True
    t1.requires_grad = True

    corr2d = Corr2D.Corr2DM( 96, padding=1, kernelSize=3, strideK=1, strideD=1 )

    # out = corr2d(t0, t0)
    # t00 = t0[:, :, 130:139, 227:236]
    # t00 = t00.contiguous()

    # rt00 = Corr2D_ext.test_from_BCHW_2_BHWC_padded(t00, 1)

    # print(rt00[0, :, :, 0])

    # L00 = Corr2D_ext.test_create_L(rt00, 3)

    # print(L00[0, 0, :, :])

    out = corr2d(t0, t0)

    print("out.min() = {}. ".format( out.min() ))
    print("out.max() = {}. ".format( out.max() ))

    idx = 96-96

    print(out[0, idx, :, :])

    print("out[0, idx, :, :].sum() = {}. ".format( out[0, idx, :, :].sum() ))
    print("%f. " % ( 540*(960-96) ))
    print("out[0, idx, :, :].mean() = {}. ".format( out[0, idx, :, :].mean() ))

    print("out[0, :, 128, 32] = \n{}".format( out[0, :, 128, 32] ))
    print("out[0, :, 128, 128] = \n{}".format( out[0, :, 128, 128] ))

    pltImg = out[0, idx, :, :].detach().cpu().numpy()

    plt.imshow( pltImg )
    plt.show()

if __name__ == "__main__":
    # torch.cuda.set_device(6)
    # test_real_image_shift(0)
    
    # torch.autograd.gradcheck()
    test_gradcheck()

    # # Load two images.
    # img0 = cv2.imread("/home/yaoyu/temp/SceneFlowSample/FlyingThings3D/RGB_cleanpass/left/0006.png", cv2.IMREAD_UNCHANGED)
    # img1 = cv2.imread("/home/yaoyu/temp/SceneFlowSample/FlyingThings3D/RGB_cleanpass/right/0006.png", cv2.IMREAD_UNCHANGED)

    # img0 = cv2.resize(img0, (16, 16), interpolation=cv2.INTER_LINEAR)
    # img1 = cv2.resize(img1, (16, 16), interpolation=cv2.INTER_LINEAR)

    # # Convert the images into tensors.
    # t0 = torch.from_numpy(img0).double().permute((2, 0, 1)).unsqueeze(0).cuda()
    # t1 = torch.from_numpy(img1).double().permute((2, 0, 1)).unsqueeze(0).cuda()

    # t0.requires_grad = True
    # t1.requires_grad = True

    # print("t0 = \n{}".format(t0))
    # print("t1 = \n{}".format(t1))

    # corr2d = Corr2D.Corr2DM( 4, padding=1, kernelSize=3, strideK=1, strideD=1 )
    
    # # Check gradient.
    # test = torch.autograd.gradcheck( corr2d, ( t0, t1 ), eps=1, atol=1e-100 )
    # print(test)





    # a = np.array( range(256, 0, -1) )
    # b = np.stack( (a, a, a), axis=0 )

    # b_0 = b[0:3, 0:3]
    # b_1 = b[0:3, 63:66]

    # print("b[0:3, 0:3] = \n{}".format( b_0 ))
    # print("b[0:3, 63:66] = \n{}".format( b_1 ))

    # zncc = zero_normalized_cross_correlation(b_0, b_1)
    # print("zncc = \n{}".format(zncc))

    # t0 = torch.from_numpy( b ).double().unsqueeze(0).unsqueeze(0)
    # t1 = t0.clone()

    # t0 = t0.cuda()
    # t1 = t1.cuda()

    # # Set the requires_grad flag.
    # t0.requires_grad = True
    # t1.requires_grad = True

    # # Test two indentical tensors.
    # cost = test_two_tensors(t0, t1)

    # costCPU = cost.detach().cpu().numpy()

    # # import ipdb; ipdb.set_trace()

    # print(costCPU[0, :, 1, 0])

    # plt.plot( costCPU[0, :, 1, 0], "-*" )
    # plt.show()
