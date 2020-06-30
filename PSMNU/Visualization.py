
import colorcet as cc
import cv2
from numba import cuda
import numpy as np

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def convert_colorcet_2_array(ccArray):
    cmap = np.zeros((256, 3), dtype=np.uint8)

    for i in range(len(ccArray)):
        rgb = hex_to_RGB( ccArray[i] )
        cmap[i, :] = rgb

    return cmap

def convert_not_finite_values(x, v):
    mask = np.logical_not( np.isfinite(x) )

    x = x.copy()
    x[mask] = v

    return x, mask

@cuda.jit
def k_convert_float_2_rgb(data, cmap, m0, m1, outImg):
    """
    data: (H, W, 1)
    cmap: (-1, 3)
    limits: 2-element array
    outImg: (H, W, 3)
    """

    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    xStride = cuda.blockDim.x * cuda.gridDim.x
    yStride = cuda.blockDim.y * cuda.gridDim.y

    steps = cmap.shape[0] - 1
    span  = m1 - m0

    for y in range( ty, data.shape[0], yStride ):
        for x in range( tx, data.shape[1], xStride ):
            value = data[y, x]

            idx = (value - m0) / span * steps

            if ( idx < 0 ):
                idx = 0
            elif ( idx > steps ):
                idx = steps

            i = int(idx)

            outImg[y, x, 0] = cmap[i, 2] # Blue.
            outImg[y, x, 1] = cmap[i, 1] # Green.
            outImg[y, x, 2] = cmap[i, 0] # Red.

def float_2_rgb(data, cmap, limits):
    """
    data is a single channel image as a NumPy array.
    """

    assert( data.ndim == 2 )

    # Dimensions.
    H, W = data.shape

    # The output image.
    img = np.zeros((H, W, 3), dtype=np.uint8)

    dData = cuda.to_device(data)
    dCMap = cuda.to_device(cmap)
    dImg  = cuda.to_device(img)

    # CUDA threads dimensions.
    CUDA_THREADS = 16
    gridX = int( np.floor(W / CUDA_THREADS) )
    gridY = int( np.floor(H / CUDA_THREADS) )

    # CUDA execution.
    cuda.synchronize()
    k_convert_float_2_rgb[[gridX, gridY, 1], [CUDA_THREADS, CUDA_THREADS, 1]](dData, dCMap, limits[0], limits[1], dImg)
    cuda.synchronize()

    img = dImg.copy_to_host()

    return img

def diff_statistics(diff):
    diff = np.abs(diff)

    return diff.mean(), diff.std()

def add_string_line_2_img(img, s, hRatio, bgColor=(70, 30, 10), fgColor=(0,255,255)):
    """
    Add a string on top of img. 
    s: The string. Only supports 1 line string.
    hRatio: The non-dimensional height ratio for the font.
    """

    assert( hRatio < 1 and hRatio > 0 )

    H, W = img.shape[:2]

    # The text size of the string in base font.
    strSizeOri, baselineOri = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)

    # The desired height.
    hDesired = np.ceil( hRatio * H )
    strScale = hDesired / strSizeOri[1]

    # The actual font size.
    strSize, baseline = \
        cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, strScale, 1 )

    fontSize, baseLineFont = \
        cv2.getTextSize("a", cv2.FONT_HERSHEY_SIMPLEX, strScale, 1 )

    # Draw the box.
    hBox = strSize[1] + fontSize[1]
    wBox = strSize[0] + 2 * fontSize[0]
    
    img = img.copy()
    pts = np.array([[0,0],[wBox,0],[wBox,hBox],[0,hBox]],np.int32)
    cv2.fillConvexPoly( img, pts, color=bgColor )
    cv2.putText(img, s, (fontSize[0], int(hBox-fontSize[1]/2.0)), 
        cv2.FONT_HERSHEY_SIMPLEX, strScale, color=fgColor, thickness=1)

    return img

def mask_rgb_img(img, mask):
    img[mask] = np.zeros(3)

def visualize_results_with_true_disparity( \
    img0, img1, trueDisp, disp0, uncertainty, trueDispMask=None, writeSize=None):
    """
    writeSize is (H, w). Set writeSize for individual images. 
    If writeSize is None, the original image size will be used.
    """

    assert( 2 == trueDisp.ndim )
    assert( 2 == disp0.ndim )

    if ( 2 == img0.ndim):
        img0 = np.stack((img0, img0, img0), axis=-1)
        img1 = np.stack((img1, img1, img1), axis=-1)
    elif ( 3 == img0.ndim ):
        if ( 1 == img0.shape[2] ):
            img0 = np.concatenate((img0, img0, img0), axis=2)
            img1 = np.concatenate((img1, img1, img1), axis=2)

    # Get the color map arrays.
    cmapDisp = convert_colorcet_2_array(cc.rainbow)
    cmapDiff = convert_colorcet_2_array(cc.coolwarm)
    cmapUnct = convert_colorcet_2_array(cc.CET_L19)

    # Image dimension.
    H, W = trueDisp.shape

    # Convert the true disparity.
    trueDisp, nfMask = convert_not_finite_values(trueDisp, 0)
    fMask = np.logical_not(nfMask) # Mask for the finite values.

    limits = [ trueDisp[fMask].min(), trueDisp[fMask].max() ]
    trueDispImg = float_2_rgb(trueDisp, cmapDisp, limits)
    disp0Img    = float_2_rgb(disp0,    cmapDisp, limits)

    # Mask trueDispImg.
    if ( trueDispMask is not None ):
        validMask = np.logical_and( fMask, trueDispMask )
    else:
        validMask = fMask

    mask_rgb_img(trueDispImg, np.logical_not(validMask))

    diffDisp0 = disp0 - trueDisp
    diffDisp0Img = float_2_rgb( diffDisp0, cmapDiff, [ -50, 50 ] )

    # The statistics.
    diffDisp0Stat = diff_statistics(diffDisp0[validMask])

    # Statistics string.
    strDiffDisp0Stat = "A: %.3f, S: %.3f" % ( diffDisp0Stat[0], diffDisp0Stat[1] )
    diffDisp0Img = add_string_line_2_img(diffDisp0Img, strDiffDisp0Stat, 0.03)

    # The uncertainty.
    maxUnct = uncertainty.max()
    minUnct = uncertainty.min()
    uncertaintyImg = float_2_rgb( uncertainty, cmapUnct, [ minUnct, maxUnct ] )
    strUnct = "min: %f, max: %f" % ( minUnct, maxUnct )
    uncertaintyImg = add_string_line_2_img( uncertaintyImg, strUnct, 0.03 )

    # The single image.
    img = np.zeros(( 2*H, 3*W, 3 ), dtype=np.uint8)

    img[0:H, 0:W]   = img0
    img[H:,  0:W]   = img1
    img[0:H, W:2*W] = trueDispImg
    img[H:,  W:2*W] = disp0Img
    img[0:H, 2*W:]  = diffDisp0Img
    img[H:,  2*W:]  = uncertaintyImg

    if ( writeSize is not None ):
        img = cv2.resize(img, (writeSize[1], writeSize[0]), interpolation=cv2.INTER_LINEAR)

    return img, diffDisp0Stat
