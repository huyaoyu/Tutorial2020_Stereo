
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
from PLYHelper import write_PLY
from Visualization import visualize_results_with_true_disparity

def read_cases(fn):
    cases = None
    
    with open(fn, 'r') as fp:
        cases = json.load(fp)
    
    if ( cases is None ):
        raise Exception("Failed to read any cases. ")

    return cases

def permute_image(img):
    if ( 3 == img.ndim ):
        if ( 3 == img.shape[2] ):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def b2mb(x): return int(x/2**20)

class TorchTracemalloc():

    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used   = b2mb(self.end-self.begin)
        self.peaked = b2mb(self.peak-self.begin)
        print(f"GPU memory end/peak usage (MB): {self.used:4d}/{self.peaked:4d}")

class Predictor(object):
    def __init__(self, name='Default'):
        super(Predictor, self).__init__()

        self.name = name
        self.model = None # The actual model.

        self.flagGray = False
        self.flagCuda = True

        self.sizeBase = 32
        self.oriSize = None # (H, W)

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

    def load_sample_data(self, fn0, fn1, fnD, resize=None):

        sampleDict = SampleLoader.load_sample(fn0, fn1, fnD, flagGray=self.flagGray, resize=resize)

        H, W = sampleDict['t0'].size()[2:4]
        self.oriSize = ( sampleDict['H'], sampleDict['W'] )

        startIdxH, endIdxH, startIdxW, endIdxW = self.find_crop_indices(H, W)

        # Crop to correct size.
        sampleDict[ 'img0'] = sampleDict[ 'img0'][startIdxH:endIdxH, startIdxW:endIdxW]
        sampleDict[ 'img1'] = sampleDict[ 'img1'][startIdxH:endIdxH, startIdxW:endIdxW]
        sampleDict[   't0'] = sampleDict[   't0'][:, :, startIdxH:endIdxH, startIdxW:endIdxW]
        sampleDict[   't1'] = sampleDict[   't1'][:, :, startIdxH:endIdxH, startIdxW:endIdxW]

        if ( sampleDict['disp0'] is not None ):
            sampleDict['disp0'] = sampleDict['disp0'][:, :, startIdxH:endIdxH, startIdxW:endIdxW]

        if ( self.flagCuda ):
            sampleDict['t0'] = sampleDict['t0'].cuda()
            sampleDict['t1'] = sampleDict['t1'].cuda()

            if ( sampleDict['disp0'] is not None ):
                sampleDict['disp0'] = sampleDict['disp0'].cuda()

        return sampleDict

    def scale_q(self, Q, qf, oriSize, resize=None):
        """oriSize is (H, W), resize is (H, W), Q is 4x4 NumPy array. """

        if ( resize is not None ):
            f = 1.0 * resize[1] / oriSize[1]
            sampleSize = resize
        else:
            f = 1.0
            sampleSize = oriSize

        startIdxH, _, startIdxW, _ = self.find_crop_indices( sampleSize[0], sampleSize[1] )

        Q = Q.copy()

        # Scale.
        Q[0, 3] *= f * qf # cx.
        Q[1, 3] *= f * qf # cy.
        Q[2, 3] *= f * qf # focal length.

        # Crop.
        Q[0, 3] += startIdxW # cx.
        Q[1, 3] += startIdxH # cy.

        return Q

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
            with TorchTracemalloc() as tt:
                startTime = time.time()

                disp0, disp1, disp2, disp3, disp4, disp5 \
                    = self.model(sample['t0'], sample['t1'], torch.Tensor([0]), torch.Tensor([0]))

                endTime = time.time()
        
            print("Predict in %fs. Size HxW: %dx%d. " % \
                ( endTime - startTime, sample['t0'].size(2), sample['t0'].size(3) ))

        return disp0.squeeze(0).squeeze(0)

    def draw(self, sampleDict, pred, fn=None):
        img0  = sampleDict['img0']
        img1  = sampleDict['img1']
        pred  = pred.cpu()

        if ( sampleDict['disp0'] is not None ):
            disp0 = sampleDict['disp0'].squeeze(0).squeeze(0).cpu().numpy()
            dispMax = disp0.max()
            dispMin = disp0.min()
        else:
            dispMax = pred.max()
            dispMin = pred.min()
            disp0 = None

        fig = plt.figure()
        ax = fig.add_subplot(2,2,1)
        plt.tight_layout()
        ax.axis('off')
        ax.set_title('Ref.')
        ax.imshow(permute_image(img0))

        ax = fig.add_subplot(2,2,3)
        # plt.tight_layout()
        ax.axis('off')
        ax.set_title('Tst.')
        ax.imshow(permute_image(img1))

        if ( disp0 is not None ):
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

        if ( fn is not None ):
            fig.savefig(fn)

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

        # Check if we have true disparity.
        if 'fnD' in caseDict.keys():
            fnD = caseDict['fnD']
        else:
            fnD = None

        # Check resize.
        if 'resize' in caseDict.keys():
            resize = caseDict['resize']
        else:
            resize = None

        sample = self.load_sample_data( 
            caseDict['fn0'], caseDict['fn1'], fnD, resize=resize)
        pred   = self.predict(sample)

        if ( fnD is not None ):
            self.visualize(sample, pred, resultFn)
        else:
            self.draw(sample, pred, resultFn)

        # Point cloud.
        if ( 'fnQ' in caseDict.keys() ):
            # Load Q.
            Q = np.loadtxt( caseDict['fnQ'], dtype=np.float32 )
            Q = self.scale_q(Q, caseDict['QF'], self.oriSize, resize)

            # The disparity offset.
            dispOff = pred.cpu().numpy() + caseDict['dOffs']

            # Get the left image.
            imgLeft = cv2.cvtColor( sample['img0'], cv2.COLOR_BGR2RGB )

            # Write PLY file.
            parts = Filesystem.get_filename_parts(resultFn)
            plyFn = '%s/%s.ply' % (parts[0], parts[1])
            write_PLY( plyFn, dispOff, Q, flagFlip=True, \
                distLimit=caseDict['distLimit'], color=imgLeft )
            
            print('Point cloud saved to %s. ' % (plyFn))


    def __str__(self):
        return '{}: flagGray={}, flagCuda={}. '.format( \
            self.name, self.flagGray, self.flagCuda )

def main():
    print("Local test the correlation disparity model. ")
    plt.close('all')

    # Read cases.
    cases = read_cases('./Cases.json')['cases']

    # Create the predictor.
    predictorC = Predictor('CorrColor')
    predictorC.load_model('./PreTrained/ERFFK1C_01_PWCNS_00.pkl', flagGray=False, kernalSize=1)
    print(predictorC)
    
    # Predict.
    for case in cases:
        if ( not case['enable'] ):
            continue

        print('>>>(color) %s' % ( case['fn0']) )
        resultFn = '%s_C.png' % ( case['name'] )
        predictorC(case, resultFn)

    del predictorC
    torch.cuda.empty_cache()

    # Create a grayscale predictor.
    predictorG = Predictor('CorrGray')
    predictorG.load_model('./PreTrained/ERFFK1_01_PWCNS_00.pkl', flagGray=True, kernalSize=1)
    print(predictorG)

    # Predict.
    for case in cases:
        if ( not case['enable'] ):
            continue

        print('>>>(Grayscale) %s' % case['fn0'])
        resultFn = '%s_G.png' % ( case['name'] )
        predictorG(case, resultFn)

    print('Done. ')

    return 0

if __name__ == '__main__':
    sys.exit(main())