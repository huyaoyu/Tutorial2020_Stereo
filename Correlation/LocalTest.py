
import cv2
import json
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

def read_cases(fn):
    cases = None
    
    with open(fn, 'r') as fp:
        cases = json.load(fp)
    
    if ( cases is None ):
        raise Exception("Failed to read any cases. ")

    return cases

class Predictor(object):
    def __init__(self, name='Default'):
        super(Predictor, self).__init__()

        self.name = name
        self.model = None # The actual model.

        self.flagGray = False
        self.flagCuda = True

        self.sizeBase = 32

    def find_closest(self, x):
        return x//self.sizeBase * self.sizeBase
    
    def find_crop_indices(self, H, W):
        # Find the closest multiple of self.sizeBase.
        h = self.find_closest(H)
        w = self.find_closest(W)

        assert(h > 2)
        assert(w > 2)

        startIdxH = int( max( ( H - h ) // 2 - 1, 0 ) )
        endIdxH   = startIdxH + h # One pass the end.

        startIdxW = int( max( ( W - w ) // 2 - 1, 0 ) )
        endIdxW   = startIdxW + w # One pass the end.

        return startIdxH, endIdxH, startIdxW, endIdxW

    def load_sample_data(self, fn0, fn1, fnD):

        sampleDict = SampleLoader.load_sample(fn0, fn1, fnD, flagGray=self.flagGray)

        H, W = sampleDict['t0'].size()[2:4]

        startIdxH, endIdxH, startIdxW, endIdxW = self.find_crop_indices(H, W)

        # Crop to correct size.
        sampleDict[ 'img0'] = sampleDict[ 'img0'][startIdxH:endIdxH, startIdxW:endIdxW]
        sampleDict[ 'img1'] = sampleDict[ 'img1'][startIdxH:endIdxH, startIdxW:endIdxW]
        sampleDict[   't0'] = sampleDict[   't0'][:, :, startIdxH:endIdxH, startIdxW:endIdxW]
        sampleDict[   't1'] = sampleDict[   't1'][:, :, startIdxH:endIdxH, startIdxW:endIdxW]
        sampleDict['disp0'] = sampleDict['disp0'][:, :, startIdxH:endIdxH, startIdxW:endIdxW]

        if ( self.flagCuda ):
            sampleDict['t0'] = sampleDict['t0'].cuda()
            sampleDict['t1'] = sampleDict['t1'].cuda()
            sampleDict['disp0'] = sampleDict['disp0'].cuda()

        return sampleDict

    def load_model(self, fn, flagGray=False, maxDisp=4, kernalSize=1):
        self.flagGray = flagGray

        params = ModelParams()
        params.set_max_disparity(maxDisp)
        params.corrKernelSize = kernalSize
        params.amp = 1
        params.flagGray = self.flagGray

        corrDispModel = CorrDispModel(params)
        SampleLoader.load_model(corrDispModel, fn)

        corrDispModel = torch.nn.DataParallel(corrDispModel)

        if ( self.flagCuda ):
            corrDispModel.cuda()

        self.model = corrDispModel

    def predict( self, sample ):
        self.model.eval()
        
        with torch.no_grad():
            startTime = time.time()

            disp0, disp1, disp2, disp3, disp4, disp5 \
                = self.model(sample['t0'], sample['t1'], torch.Tensor([0]), torch.Tensor([0]))

            endTime = time.time()
        
            print("Predict in %fs. Size HxW: %dx%d. " % \
                ( endTime - startTime, sample['t0'].size(2), sample['t0'].size(3) ))

        return disp0.squeeze(0).squeeze(0)

    def draw(self, sampleDict, pred ):
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

    def visualize( self, sampleDict, pred, fn=None ):
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

    def __call__(self, caseDict, resultFn):
        if ( self.model is None ):
            raise Exception("Must load model first. ")

        sample = self.load_sample_data( 
            caseDict['fn0'], caseDict['fn1'], caseDict['fnD'])
        pred   = self.predict(sample)
        # self.draw(sample, pred)
        self.visualize(sample, pred, resultFn)

    def __str__(self):
        return '{}: flagGray={}, flagCuda={}. '.format( \
            self.name, self.flagGray, self.flagCuda )

def main():
    print("Local test the correlation disparity model. ")
    plt.close('all')

    # Read cases.
    cases = read_cases('./Cases.json')['cases']

    flagGray = False

    # Create the predictor.
    predictorC = Predictor('CorrColor')
    predictorC.load_model('./PreTrained/ERFFK1C_01_PWCNS_00.pkl', flagGray=False, kernalSize=1)
    print(predictorC)
    
    # Predict.
    for case in cases:
        print('>>>(color) %s' % ( case['fn0']) )
        resultFn = '%s_C.png' % ( case['name'] )
        predictorC(case, resultFn)

    # Create a grayscale predictor.
    predictorG = Predictor('CorrGray')
    predictorG.load_model('./PreTrained/ERFFK1_01_PWCNS_00.pkl', flagGray=True, kernalSize=1)
    print(predictorG)

    # Predict.
    for case in cases:
        print('>>>(Grayscale) %s' % case['fn0'])
        resultFn = '%s_G.png' % ( case['name'] )
        predictorG(case, resultFn)

    print('Done. ')

    return 0

if __name__ == '__main__':
    sys.exit(main())