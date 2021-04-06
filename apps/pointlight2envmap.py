import cv2
import os
import numpy as np
import cvgutils.Utils as util
import argparse
import tqdm
import torch


def pointlight2envmap(imseq,indexfn,envmap,scale,output,w,h):
    envmap = cv2.imread(envmap,-1)
    envmap = cv2.resize(envmap,(w,h))
    h = envmap.shape[0]
    w = envmap.shape[1]
    index = util.loadPickle(indexfn)
    assert len(index) == h * w

    x = np.linspace(0,w-1,w)
    y = np.linspace(0,h-1,h)
    u = (x + 0.5) / w
    v = (y + 0.5) / h
    u, v = np.meshgrid(u,v)
    x, y = np.meshgrid(x,y)
    sintScaled = np.sin(v*np.pi) * scale
    img = cv2.imread(imseq%int(index['0000_0000']),-1)
    img = img * 0
    for x0,y0,intensity, idx in tqdm.tqdm(zip(x.reshape(-1).astype(np.uint8),y.reshape(-1).astype(np.uint8),envmap.reshape(-1,3),index.values())):
        intensity = envmap[y0,x0,None,None,:] * sintScaled[y0,x0]
        fn = os.path.join(imseq%int(index['%04d_%04d' % (x0,y0)]))
        im = cv2.imread(fn,-1)
        img += im * intensity
    cv2.imwrite(output,img.astype(np.float32))
    cv2.imwrite(output.replace('exr','png'),(np.clip(img,0,1)**(1/2.2)*255).astype(np.uint8))
    return img.astype(np.float32)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--imseq', type=str,default='/home/mohammad/Projects/NRV/dataset/buddha/testImgsExr-test-random/%07i.exr', help='List of images to combine')
parser.add_argument('--indexfn', type=str,default='/home/mohammad/Projects/NRV/dataset/buddha/testData-test-random/index.pickle', help='index of images')
parser.add_argument('--envmap', type=str,default='/home/mohammad/Projects/NRV/NrArtFree/cvgutils/tests/testimages/uffizi-large.exr', help='Filename of the environment map')
parser.add_argument('--output', type=str,default='/home/mohammad/Projects/NRV/dataset/buddha/3x1-gt.exr', help='Filename of the environment map')
opt = parser.parse_args()

pointlight2envmap(opt.imseq,opt.indexfn,opt.envmap,10,opt.output,3,1)
# torch.save(opt,'opt.pickle')
# opt = torch.load('opt.pickle')