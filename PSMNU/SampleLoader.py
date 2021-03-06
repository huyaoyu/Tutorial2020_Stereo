import cv2
import numpy as np
import os

import torch
import torchvision

from CommonPython.Filesystem import Filesystem

from IO import readPFM

def test_file(fn):
    if ( not os.path.isfile(fn) ):
        raise Exception("File %s not exist. " % (fn))

def load_image(fn, resize=None):
    test_file(fn)

    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

    H, W = img.shape[0:2]
    
    if ( resize is not None ):
        img = cv2.resize(img, ( resize[1], resize[0] ), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray, H, W

def convert_2_tensor(img):
    tt = torchvision.transforms.ToTensor()

    tImg = tt(img)

    if ( 2 == tImg.dim() ):
        return tImg.unsqueeze(0)
    else:
        return tImg

def load_disp(fn):
    test_file(fn)

    # Get the ext of the filename.
    parts = Filesystem.get_filename_parts(fn)

    ext = parts[2].lower()

    if ( '.pfm' == ext ):
        disp, scale = readPFM(fn)
    elif ( '.npy' == ext ):
        disp = np.load(fn).astype(np.float32)
    else:
        raise Exception("Not supported ext {}. ".format(ext))

    return disp

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

class NormalizeRGB_OCV(object):
    def __init__(self, s=1.0/255):
        super(NormalizeRGB_OCV, self).__init__()
        
        self.s = s

    def __call__(self, x):
        """This is the OpenCV version. The order of the color channle is BGR. The order of dimension is HWC."""

        x = np.copy(x) * self.s

        # It is assumed that the data type of x is already floating point number.
        # Note the order of channels. OpenCV uses BGR.
        x[:, :, 0] = ( x[:, :, 0] - imagenet_stats["mean"][2] ) / imagenet_stats["std"][2]
        x[:, :, 1] = ( x[:, :, 1] - imagenet_stats["mean"][1] ) / imagenet_stats["std"][1]
        x[:, :, 2] = ( x[:, :, 2] - imagenet_stats["mean"][0] ) / imagenet_stats["std"][0]

        return x

def load_sample(fn0, fn1, disp0=None, flagGray=False, resize=None):
    # Load the image by OpenCV
    img0, gray0, H, W = load_image(fn0, resize=resize)
    img1, gray1, _, _ = load_image(fn1, resize=resize)
    
    # Convert the images to PyTorch tensors.
    if ( flagGray ):
        t0 = convert_2_tensor(gray0.astype(np.float32) / 255.0)
        t1 = convert_2_tensor(gray1.astype(np.float32) / 255.0)
    else:
        t0 = convert_2_tensor(img0)
        t1 = convert_2_tensor(img1)

        normalizer = torchvision.transforms.Normalize(**imagenet_stats)
        
        t0 = normalizer(t0)
        t1 = normalizer(t1)

    # Make dummy mini-batch.
    t0 = t0.unsqueeze(0)
    t1 = t1.unsqueeze(0)
    
    # Load the disparity.
    if ( disp0 is not None ):
        disp = load_disp(disp0)
        td = convert_2_tensor(disp.astype(np.float32))
        td = td.unsqueeze(0)
    else:
        td = None

    if ( flagGray ):
        return { \
            'img0': gray0, 'img1': gray1, \
            't0': t0, 't1': t1, \
            'disp0': td, \
            'H': H, 'W': W }
    else:
        return { \
            'img0': img0, 'img1': img1, \
            't0': t0, 't1': t1, \
            'disp0': td, 
            'H': H, 'W': W }

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
        print("DataParallel version OK.")

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

    sampleDictC = load_sample(fn0, fn1, fnD, flagGray=False)

    # Show the dimensions of loaded sample.
    print(sampleDictC['t0'].size())

    sampleDictG = load_sample(fn0, fn1, fnD, flagGray=True)
    # Show the dimensions of loaded sample.
    print(sampleDictG['t0'].size())

