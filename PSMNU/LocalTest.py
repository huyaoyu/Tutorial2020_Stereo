
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

import torch

from CommonPython.Filesystem import Filesystem

from Model.PyramidNet import PSMNetWithUncertainty as PSMNU
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

def load_model(flagCuda=True):
    psmnu = PSMNU(1, 32, 256)
    SampleLoader.load_model(psmnu, "./PreTrained/PU_03_PSMNet_02.pkl")

    psmnu = torch.nn.DataParallel(psmnu)

    if ( flagCuda ):
        psmnu.cuda()

    return psmnu

def predict( model, sample ):
    model.eval()
    
    with torch.no_grad():
        startTime = time.time()

        output3, logSigSqu = model(sample["t0"], sample["t1"])

        endTime = time.time()

        sig = torch.exp( logSigSqu / 2.0 )
    
        print("Predict in %fs. " % ( endTime - startTime ))

    pred = output3.squeeze(0).squeeze(0)
    sig  = sig.squeeze(0).squeeze(0)

    return pred, sig

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

def visualize( sampleDict, pred, sig, fn=None ):
    img0  = sampleDict['img0']
    img1  = sampleDict['img1']
    disp0 = sampleDict['disp0'].squeeze(0).squeeze(0).cpu().numpy()
    pred  = pred.cpu().numpy()
    sig   = sig.cpu().numpy()

    visImg, diffStat = visualize_results_with_true_disparity( \
        img0, img1, disp0, pred, sig )

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
    print("Local test the PSMNU model. ")
    flagGray = True
    psmnu  = load_model()
    sample = load_sample_data(flagGray=flagGray)
    pred, sig = predict(psmnu, sample)
    # draw(sample, pred)
    visualize(sample, pred, sig, 'VisResultUnct.png')

    return 0

if __name__ == '__main__':
    sys.exit(main())