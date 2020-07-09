
# Author
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>

import copy
import numpy as np
from plyfile import PlyData, PlyElement

def convert_2D_array_2_1D_list(a):
    """
    Return a 2D NumPy array into a list of tuples.
    """

    res = []

    for i in range(a.shape[0]):
        t = tuple( a[i,:].tolist() )
        res.append( t )
    
    return res

def write_PLY(fn, disp, Q, flagFlip=False, distLimit=100., mask=None, color=None, binary=True):
    """
    fn: The output filename.
    disp: The disparity. A NumPy array with dimension (H, W).
    mask: Logical NumPy array.
    Q: The reprojection matrix. A NumP array of dimension 4x4.
    color: The color image. The image could be (H, W) or (H, W, C).
           C could be 1 or 3. If color==None, no color properties 
           will be in the output PLY file.
    binary: Set True to write a binary format PLY file. Set False for
            a ASCII version.
    """

    disp = disp.copy()

    # Get the size of the image.
    H = disp.shape[0]
    W = disp.shape[1]

    # Make x and y.
    xLin = np.linspace( 0, W-1, W, dtype=np.float32 )
    yLin = np.linspace( 0, H-1, H, dtype=np.float32 )

    x, y = np.meshgrid( xLin, yLin )
    x = x.reshape((-1,))
    y = y.reshape((-1,))

    # Make the coordinate array.
    d  = disp.reshape((-1,))
    hg = np.ones((1, d.shape[0]), dtype=np.float32).reshape((-1,))

    # Mask.
    m  = d > 0
    nm = np.logical_not(m)
    m  = m.reshape((-1,))
    
    d[nm] = 1.0

    if ( mask is not None ):
        m = np.logical_and( m, mask.reshape((-1,)) )

    coor = np.stack( [ x, y, d, hg ], axis=-1 ).transpose()

    # Calculate the world coordinate.
    if ( flagFlip ):
        Q = Q.copy()
        Q[1, 1] *= -1
        Q[1, 3] *= -1
        Q[2, 3] *= -1

    coor = Q.dot( coor )

    coor[0, :] = coor[0, :] / coor[3, :]
    coor[1, :] = coor[1, :] / coor[3, :]
    coor[2, :] = coor[2, :] / coor[3, :]

    coor = coor.transpose()[:, 0:3]

    # Filter the points. Only keep the points within distLimit.
    dispMask = np.abs( coor[:, 2] ) <= distLimit
    m = np.logical_and( m, dispMask.reshape((-1,)) )

    # Handle color.
    if ( color is not None ):
        if ( 2 == len( color.shape ) ):
            color = np.stack([ color, color, color ], axis=-1)
        
        color = color.reshape(-1, 3)
        color = color[m, 0:3]
        coor  = coor[m, 0:3]
        
        color = np.clip( color, 0, 255 ).astype(np.uint8)

        # Concatenate.
        vertex = np.concatenate([coor, color], axis=1)

        # Create finial vetex array.
        vertex = convert_2D_array_2_1D_list(vertex)
        vertex = np.array( vertex, dtype=[\
            ( "x", "f4" ), \
            ( "y", "f4" ), \
            ( "z", "f4" ), \
            ( "red", "u1" ), \
            ( "green", "u1" ), \
            ( "blue", "u1" ) \
            ] )
    else:
        coor = convert_2D_array_2_1D_list(coor)
        vertex = np.array( coor, dtype=[\
            ( "x", "f4" ), \
            ( "y", "f4" ), \
            ( "z", "f4" ) \
            ] )
    
    # Save the PLY file.
    el = PlyElement.describe(vertex, "vertex")

    PlyData([el], text= (not binary) ).write(fn)

if __name__ == "__main__":
    print("Test write_PLY.")

    # Disparity.
    disp  = np.linspace(1, 10, 10, dtype=np.float32).reshape(2, 5)
    r = np.linspace(0, 9, 10).reshape(2, 5)
    g = np.linspace(0, 9, 10).reshape(2, 5) + 10
    b = np.linspace(0, 9, 10).reshape(2, 5) + 100
    color = np.stack([r, g, b], axis=-1)
    color = color.astype(np.uint8)

    Q = np.eye((4), dtype=np.float32)
    Q[3, 3] = 5

    print("disp=\n{}".format(disp))
    print("color\n{}".format(color))
    print("color.shape={}".format(color.shape))
    print("Q=\n{}".format(Q))

    # Write a PLY file.
    write_PLY("DispColor3NoMask.ply", disp, Q, color=color, binary=False)

    # Write a PLY file with single channel color.
    write_PLY("DispColor1NoMask.ply", disp, Q, color=color[:,:,0], binary=False)

    mask = np.logical_and( disp > 3, disp < 8 )
    print("mask={}".format(mask))

    # Write a PLY file with 3-channel color and a mask.
    write_PLY("DispColor3Mask.ply", disp, Q, mask=mask, color=color, binary=False)

    # Write a PLY file with 1-channel color and a mask.
    write_PLY("DispColor1Mask.ply", disp, Q, mask=mask, color=color[:,:,0], binary=False)
    