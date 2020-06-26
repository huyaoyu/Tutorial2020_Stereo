import cv2
import numpy as np
import os

import torch
import torchvision

from IO import readPFM

def test_file(fn):
    if ( not os.path.isfile(fn) ):
        raise Exception("File %s not exist. " % (fn))

def load_image(fn):
    test_file(fn)

    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray

def convert_2_tensor(img):
    tt = torchvision.transforms.ToTensor()

    tImg = tt(img)

    if ( 2 == tImg.dim() ):
        return tImg.unsqueeze(0)
    else:
        return tImg

def load_disp(fn):
    test_file(fn)

    disp, scale = readPFM(fn)

    return disp

def load_sample(fn0, fn1, disp0):
    # Load the image by OpenCV
    img0, gray0 = load_image(fn0)
    img1, gray1 = load_image(fn1)
    
    # Convert the images to PyTorch tensors.
    t0 = convert_2_tensor(gray0.astype(np.float32) / 255.0)
    t1 = convert_2_tensor(gray1.astype(np.float32) / 255.0)

    # Load the disparity.
    disp = load_disp(disp0)
    td = convert_2_tensor(disp.astype(np.float32))

    # Make dummy mini-batch.
    t0 = t0.unsqueeze(0)
    t1 = t1.unsqueeze(0)
    td = td.unsqueeze(0)

    return {'img0': t0, 'img1': t1, 'disp0': td}

def load_model(model, modelname):
    preTrainDict = torch.load(modelname)
    model_dict = model.state_dict()
    preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

    if( 0 == len(preTrainDictTemp) ):
        print("Does not find any module to load. Try DataParallel version.")
        for k, v in preTrainDict.items():
            kk = k[7:]

            if ( kk in model_dict ):
                preTrainDictTemp[kk] = v

        preTrainDict = preTrainDictTemp

    if ( 0 == len(preTrainDict) ):
        raise Exception("Could not load model from %s." % (modelname), "load_model")

    # for item in preTrainDict:
    #     print("Load pretrained layer:{}".format(item) )
    model_dict.update(preTrainDict)
    model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    # Test locally.
    print("Test SampleLoader.py locally. ")

    fn0 = '../SampleData/SceneFlow_FlyingThings3D/Left/0006.png'
    fn1 = '../SampleData/SceneFlow_FlyingThings3D/Right/0006.png'
    fnD = '../SampleData/SceneFlow_FlyingThings3D/Disparity/0006.pfm'

    sampleDict = load_sample(fn0, fn1, fnD)

