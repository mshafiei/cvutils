import cv2
import numpy as np

def writePng(fn, im):
    """[Tonemap a linear image with quasi srgb (clip and gamma 1/2.2) and write in filename with png extension]

    Args:
        im ([[hxwx3 ndarray]): [linear image]
        fn (str): [filename with png extension]

    Returns:
        [hxwx3 ndarray]: [srgb image]
    """
    cv2.imwrite(fn, hdr2srgb(im))

def hdr2srgb(im):
    """[Tonemap a linear image with quasi srgb (clip and gamma 1/2.2)]

    Args:
        im ([[hxwx3 ndarray]): [linear image]

    Returns:
        [hxwx3 ndarray]: [srgb image]
    """

    return (np.clip(im[:,:,::-1],0,1) ** (1/2.2) * 255).astype(np.uint8)

def resize(im,scale=None,dx=None,dy=None):
    """[Rescales an image]

    Args:
        im ([h,w,3 ndarray]): [input image]
        scale ([float], optional): [scale factor]. Defaults to None.
        dx ([int], optional): [new width]. Defaults to None.
        dy ([int], optional): [new height]. Defaults to None.

    Returns:
        [dy,dx,3 ndarray]: [rescaled image]
    """

    if(not(scale is None)):
        dsize = (int(im.shape[1]*scale),int(im.shape[0]*scale))
    elif((not(dx is None)) and (not(dy is None))):
         dsize = (dx,dy)    
    else:
        Exception("At least one of the scale parameters should be set")
    return cv2.resize(im,dsize=dsize,interpolation = cv2.INTER_AREA)