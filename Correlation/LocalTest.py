
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

import torch

from Model.PWCNetStereo import PWCNetStereoParams as ModelParams
from Model.PWCNetStereo import PWCNetStereoRes as CorrDispModel
import SampleLoader

def load_sample_data(flagCuda=True):
    fn0 = '../SampleData/SceneFlow_FlyingThings3D/Left/0006.png'
    fn1 = '../SampleData/SceneFlow_FlyingThings3D/Right/0006.png'
    fnD = '../SampleData/SceneFlow_FlyingThings3D/Disparity/0006.pfm'

    sampleDict = SampleLoader.load_sample(fn0, fn1, fnD)

    H = sampleDict['img0'].size(2)

    if ( H <= 512 ):
        raise Exception("Image height must be larger than 512 pixels. ")

    startIdx = int( max( ( H - 512 ) // 2 - 1, 0 ) )
    endIdx   = startIdx + 512 # One pass the end.

    # Crop to correct size.
    sampleDict[ 'img0'] = sampleDict[ 'img0'][:, :, startIdx:endIdx, :]
    sampleDict[ 'img1'] = sampleDict[ 'img1'][:, :, startIdx:endIdx, :]
    sampleDict['disp0'] = sampleDict['disp0'][:, :, startIdx:endIdx, :]

    if ( flagCuda ):
        sampleDict['img0'] = sampleDict['img0'].cuda()
        sampleDict['img1'] = sampleDict['img1'].cuda()
        sampleDict['disp0'] = sampleDict['disp0'].cuda()

    return sampleDict

def load_model(flagCuda=True):
    params = ModelParams()
    params.set_max_disparity(4)
    params.corrKernelSize = 3
    params.amp = 1
    params.flagGray = True

    corrDispModel = CorrDispModel(params)
    SampleLoader.load_model(corrDispModel, "./PreTrained/ERFFK3_01_PWCNS_00.pkl")

    corrDispModel = torch.nn.DataParallel(corrDispModel)

    if ( flagCuda ):
        # corrDispModel.set_cpu_mode()
        corrDispModel.cuda()

    return corrDispModel

def predict( model, sample ):
    model.eval()
    
    with torch.no_grad():
        disp0, disp1, disp2, disp3, disp4, disp5 \
            = model(sample["img0"], sample["img1"], torch.Tensor([0]), torch.Tensor([0]))

    return disp0.squeeze(0).squeeze(0)

def draw( sampleDict, pred ):
    img0  = sampleDict['img0'].squeeze(0).squeeze(0).cpu().numpy()
    img1  = sampleDict['img1'].squeeze(0).squeeze(0).cpu().numpy()
    disp0 = sampleDict['disp0'].squeeze(0).squeeze(0).cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    plt.tight_layout()
    ax.axis('off')
    ax.set_title('Ref.')
    ax.imshow(img0)

    ax = fig.add_subplot(2,2,3)
    # plt.tight_layout()
    ax.axis('off')
    ax.set_title('Tst.')
    ax.imshow(img1)

    dispMax = disp0.max()
    dispMin = disp0.min()

    ax = fig.add_subplot(2,2,2)
    # plt.tight_layout()
    ax.axis('off')
    ax.set_title('disp0')

    disp0 = disp0 - dispMin
    disp0 = disp0 / ( dispMax - dispMin )

    ax.imshow(disp0)

    ax = fig.add_subplot(2,2,4)
    # plt.tight_layout()
    ax.axis('off')
    ax.set_title('pred')

    pred = pred - dispMin
    pred = pred / ( dispMax - dispMin )

    ax.imshow(pred)

    plt.show()
    plt.close(fig)

def main():
    print("Local test the correlation disparity model. ")
    corrDisp = load_model()
    sample   = load_sample_data()
    pred     = predict(corrDisp, sample)
    pred     = pred.cpu()
    draw(sample, pred)

    return 0

if __name__ == '__main__':
    sys.exit(main())