import cv2
import numpy as np
import torch
import OpenEXR as exr
import tqdm
def writePng(fn, im):
    """[Tonemap a linear image with quasi srgb (clip and gamma 1/2.2) and write in filename with png extension]

    Args:
        im ([[hxwx3 ndarray]): [linear image]
        fn (str): [filename with png extension]

    Returns:
        [hxwx3 ndarray]: [srgb image]
    """
    if(type(im[0,0,0]) == np.uint8):
        cv2.imwrite(fn, im[:,:,::-1])
    else:
        cv2.imwrite(fn, hdr2srgb(im)[:,:,::-1])

def hdr2srgb(im):
    """[Tonemap a linear image with quasi srgb (clip and gamma 1/2.2) \in [0-1]]

    Args:
        im ([[hxwx3 ndarray]): [linear image]

    Returns:
        [hxwx3 ndarray]: [srgb image \in [0,255]]
    """
    if(type(im) == torch.Tensor):
        im = im.cpu().numpy()
    
    return (np.clip(im,0,1) ** (1/2.2) * 255).astype(np.uint8)

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

def imageseq2avi(fn,ims,enlarge=True,fps=30,Tonemap=True):
    """[summary]

    Args:
        fn (str): [video filename .avi]
        ims ([n x c x h x w ndarray or list of strs]): [Array or tensor or list of filenames containing n image frames \in 0-1]
        fps (int, optional): [Frames per second]. Defaults to 30.
    """
    if(type(ims) == torch.Tensor):
        ims = ims.cpu().numpy()
        out = cv2.VideoWriter(fn,cv2.VideoWriter_fourcc(*'DIVX'), fps, (ims.shape[3],ims.shape[2]))
    elif(type(ims) == list):
        im = cv2.imread(ims[0],-1)
        out = cv2.VideoWriter(fn,cv2.VideoWriter_fourcc(*'DIVX'), fps, (im.shape[1],im.shape[0]))
    elif(type(ims) == np.ndarray):
            out = cv2.VideoWriter(fn,cv2.VideoWriter_fourcc(*'DIVX'), fps, (ims.shape[3],ims.shape[2]))
    else:
        raise "didn't recognize input type"
    print('Writing video file')
    for img in tqdm.tqdm(ims):
        if(type(ims) == list):#it's a list of filenames
            im = cv2.imread(img,-1)
            if(enlarge):
                im = cv2.resize(im,(im.shape[1] * 2, im.shape[0] * 2))
            tmp = hdr2srgb(im)
            tmp = tmp
        else:
            im = img
            tmp = hdr2srgb(im)
            tmp = tmp.transpose([1,2,0])[:,:,::-1]

        out.write(tmp)

    out.release()


def loadImageSeq(fns):
    """
    [In: list of image filenames, out:nxhxwx3 array of images]
    """
    print('Reading images')
    im = []
    for fn in tqdm.tqdm(fns):
        im.append(cv2.imread(fn,-1))
    return np.stack(im,axis=0)

def readChannelExr(fn):
    """[Returns a dictionary mapping channel id to corresponding hxw image. Mostly copied from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b]

    Args:
        fn (str): [Exr filename]

    Returns:
        [dict]: [dictionary mapping channel id to height x width image]
    """

    import Imath
    exrfile = exr.InputFile(fn)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    channelData = dict()
    
    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)
        
        channelData[c] = C.transpose(1,0)
    
    return channelData
    

def readExrImage(fn,channels=[['R','G','B','A']]):
    channelData = readChannelExr(fn)
    
    imgs = []
    for cset in channels:
        #make sure all required channels are in the file
        assert(np.array([cset[i] in channelData.keys() for i in range(len(cset))]).all())
        img = []
        for c in cset:
            img.append(channelData[c])
        imgs.append(np.stack(img,axis=-1).transpose(1,0,2))
    return imgs

def depth2txt(fn,d,im):
    """[Takes a hxwx3 depth image and saves it in a text file]

    Args:
        fn (str): [filename]
        d ([ndarray]): [depth image]
        im ([ndarray]): [reflectanec image]
    """
    if(not(im is None)):
        assert(im.shape == d.shape)
        out = np.concatenate((d.reshape(-1,3),im.reshape(-1,3)),axis=1)
    else:
        out = d.reshape(-1,3)
    np.savetxt(fn,out,fmt='%-10.5f')






