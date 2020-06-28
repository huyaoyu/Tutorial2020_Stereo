
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

import torch

from CommonPython.Filesystem import Filesystem

from Model.PWCNetStereo import PWCNetStereoParams as ModelParams
from Model.PWCNetStereo import PWCNetStereoRes as CorrDispModel
import SampleLoader
from Visualization import visualize_results_with_true_disparity

def load_sample_data(flagGray=False, flagCuda=True):
    fn0 = '../SampleData/SceneFlow_FlyingThings3D/Left/0006.png'
    fn1 = '../SampleData/SceneFlow_FlyingThings3D/Right/0006.png'
    fnD = '../SampleData/SceneFlow_FlyingThings3D/Disparity/0006.pfm'

    sampleDict = SampleLoader.load_sample(fn0, fn1, fnD, flagGray=flagGray)

    H = sampleDict['t0'].size(2)

    if ( H <= 512 ):
        raise Exception("Image height must be larger than 512 pixels. ")

    startIdx = int( max( ( H - 512 ) // 2 - 1, 0 ) )
    endIdx   = startIdx + 512 # One pass the end.

    # Crop to correct size.
    sampleDict[ 'img0'] = sampleDict[ 'img0'][startIdx:endIdx, :]
    sampleDict[ 'img1'] = sampleDict[ 'img1'][startIdx:endIdx, :]
    sampleDict[   't0'] = sampleDict[   't0'][:, :, startIdx:endIdx, :]
    sampleDict[   't1'] = sampleDict[   't1'][:, :, startIdx:endIdx, :]
    sampleDict['disp0'] = sampleDict['disp0'][:, :, startIdx:endIdx, :]

    if ( flagCuda ):
        sampleDict['t0'] = sampleDict['t0'].cuda()
        sampleDict['t1'] = sampleDict['t1'].cuda()
        sampleDict['disp0'] = sampleDict['disp0'].cuda()

    return sampleDict

def load_model(flagGray=False, flagCuda=True):
    params = ModelParams()
    params.set_max_disparity(4)
    params.corrKernelSize = 1
    params.amp = 1
    params.flagGray = flagGray

    corrDispModel = CorrDispModel(params)
    if (flagGray):
        SampleLoader.load_model(corrDispModel, "./PreTrained/ERFFK3_01_PWCNS_00.pkl")
    else:
        SampleLoader.load_model(corrDispModel, "./PreTrained/ERFFK1C_01_PWCNS_00.pkl")

    corrDispModel = torch.nn.DataParallel(corrDispModel)

    if ( flagCuda ):
        # corrDispModel.set_cpu_mode()
        corrDispModel.cuda()

    return corrDispModel

def predict( model, sample ):
    model.eval()
    
    with torch.no_grad():
        startTime = time.time()

        disp0, disp1, disp2, disp3, disp4, disp5 \
            = model(sample['t0'], sample['t1'], torch.Tensor([0]), torch.Tensor([0]))

        endTime = time.time()
    
        print("Predict in %fs. " % ( endTime - startTime ))

    return disp0.squeeze(0).squeeze(0)

def draw( sampleDict, pred ):
    img0  = sampleDict['img0']
    img1  = sampleDict['img1']
    disp0 = sampleDict['disp0'].squeeze(0).squeeze(0).cpu().numpy()
    pred  = pred.cpu()

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

def visualize( sampleDict, pred, fn=None ):
    img0  = sampleDict['img0']
    # img1  = sampleDict['img1']
    disp0 = sampleDict['disp0'].squeeze(0).squeeze(0).cpu().numpy()
    pred  = pred.cpu()

    visImg, diffStat = visualize_results_with_true_disparity( \
        img0, disp0, pred )

    print("Mean error = %f, std = %f. " % ( diffStat[0], diffStat[1] ))

    if ( fn is not None ):
        Filesystem.test_directory_by_filename(fn)
        cv2.imwrite( fn, visImg )
        print("Result visualization saved to %s. " % (fn))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(cv2.cvtColor(visImg, cv2.COLOR_BGR2RGB))

    plt.show()
    plt.close(fig)

def main():
    print("Local test the correlation disparity model. ")
    flagGray = False
    corrDisp = load_model(flagGray=flagGray)
    sample   = load_sample_data(flagGray=flagGray)
    pred     = predict(corrDisp, sample)
    # draw(sample, pred)
    visualize(sample, pred, 'VisResultCorr.png')

    return 0

if __name__ == '__main__':
    sys.exit(main())