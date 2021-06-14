import cv2
import os
import numpy as np
import cvgutils.Utils as util
import cvgutils.Image as Im
import argparse
import tqdm
import torch
from apps.pointlight2envmap import pointlight2envmap

def pointlight2envmapSeq(light_w,light_h,envmap_dir,imseq,output,attachEnvmap,scale):
    l = light_w * light_h
    envfns = [i for i in sorted(os.listdir(envmap_dir)) if '.exr' in i]
    if(not os.path.exists(output)):
        os.makedirs(output)
    #load images
    imgfns = [i for i in sorted(os.listdir(imseq)) if '.exr' in i]
    im = []
    print('Load images')
    for fn in tqdm.tqdm(imgfns[:l]):
        if('.exr' not in fn):
            continue
        im.append(cv2.imread(os.path.join(imseq,fn),-1))
    im = np.stack(im,axis=0)#.reshape(opt.light_h,opt.light_w,*im[0].shape).transpose([1,0,2,3,4]).reshape(-1,*im[0].shape)

    
    print('Load envmap')
    for fn in tqdm.tqdm(envfns[:l]):
        if('.exr' not in fn):
            continue
        print(fn)
        out = os.path.join(output,fn)
        env_im = cv2.imread(os.path.join(envmap_dir,fn),-1)
        relit = (env_im.reshape(-1,3)[:,None,None,:] * im).sum(0)
        if(attachEnvmap):
            env_im = cv2.resize(env_im,(env_im.shape[1]*5,env_im.shape[0]*5)) * (1/scale)
            relit[:env_im.shape[0],:env_im.shape[1],:] = env_im
        cv2.imwrite(out,relit * scale)
        cv2.imwrite(out.replace('.exr','.png'), (np.clip(relit * scale,0,1) ** (1/2.2) * 255).astype(np.uint8))

if __name__ == "__main__":
    #Takes a LTM and an envmap seq. then creates an animation
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--imseq', type=str,default='/home/mohammad/Projects/NRV/dataset/buddha/testImgsExr-test-random/', help='List of images to combine')
    parser.add_argument('--indexfn', type=str,default='/home/mohammad/Projects/NRV/dataset/buddha/testData-test-random/index.pickle', help='index of images')
    parser.add_argument('--envmap_seq', type=str,default='/home/mohammad/Projects/NRV/NrArtFree/cvgutils/tests/testimages/uffizi-large.exr', help='Filename of the environment map')
    parser.add_argument('--output', type=str,default='/home/mohammad/Projects/NRV/dataset/buddha/', help='Filename of the environment map')
    parser.add_argument('--light_w', type=int,default=24, help='Light width')
    parser.add_argument('--light_h', type=int,default=12, help='Light height')
    parser.add_argument('--scale', type=float,default=1, help='Scale of output')
    parser.add_argument('--attachEnvmap', type=int,default=1, help='Attaches the relighting envmap to top left')
    opt = parser.parse_args()
    
    pointlight2envmapSeq(opt.light_w,opt.light_h,opt.envmap_seq,opt.imseq,opt.output,opt.attachEnvmap,opt.scale)

