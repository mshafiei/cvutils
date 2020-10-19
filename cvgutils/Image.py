import cv2
import numpy as np
import torch
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
    """[Tonemap a linear image with quasi srgb (clip and gamma 1/2.2) \in [0-1]]

    Args:
        im ([[hxwx3 ndarray]): [linear image]

    Returns:
        [hxwx3 ndarray]: [srgb image \in [0,255]]
    """
    if(type(im) == torch.Tensor):
        im = im.cpu().numpy()

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

def imageseq2avi(fn,ims,fps=30,Tonemap=True):
    """[summary]

    Args:
        fn (str): [filename]
        ims ([n x c x h x w ndarray]): [Array or tensor containing n image frames \in 0-1]
        fps (int, optional): [Frames per second]. Defaults to 30.
    """
    if(type(ims) == torch.Tensor):
        ims = ims.cpu().numpy()

    out = cv2.VideoWriter(fn,cv2.VideoWriter_fourcc(*'DIVX'), fps, (ims.shape[3],ims.shape[2]))
    for im in ims:
        tmp = hdr2srgb(im)
        tmp = tmp.transpose([1,2,0])[:,:,::-1]
        out.write(tmp)

    out.release()

