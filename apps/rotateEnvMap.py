import numpy as np
import torch
import cvgutils.Linalg as lin
import cvgutils.Image as Im
import cv2
import torch.nn.functional as F
import os
import tqdm
import argparse
from PIL import Image
from cvgutils.Image import imageseq2avi, loadImageSeq, TileImages
import glob
def rotateEnvMap(im,deg):
    h,w,_ = im.shape
    u, v = np.meshgrid(np.arange(0,w)/w, np.arange(0,h)/h)
    p, t = lin.uv2pt(u,v)
    x,y,z = lin.pt2xyz(p,t)
    xyz = np.stack((x,y,z),axis=-1)
    m = lin.rotate2D(deg)
    xyz = (m * xyz[...,None,:]).sum(-1)
    p, t = lin.xyz2pt(xyz[...,0],xyz[...,1],xyz[...,2])
    u, v = lin.pt2uv(p,t)
    uv = np.stack((u*2-1,v*2-1),axis=-1)
    im = torch.from_numpy(im[None,...]).permute(0,3,1,2)
    uv = torch.from_numpy(uv[None,...]).reshape(1,h,w,2)
    return F.grid_sample(im,uv.float()).permute(0,2,3,1)[0].cpu().numpy()

def rotateEnvmapSequence(outdir, envmap_fn, nimgs, env_w, env_h):
    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    im = cv2.imread(envmap_fn,-1)
    im = Im.antialiasResize(im,im.shape[1]//8,im.shape[0]//8)
    print('writing images to ', outdir)
    for i in tqdm.trange(nimgs):
        r = float(i) / nimgs * 360
        im2 = rotateEnvMap(im,r)
        im2 = Im.antialiasResize(im2,env_w,env_h)
        # im2 = (np.clip(im2,0,1) ** (1/2.2) * 255).astype(np.uint8)
        outfn = os.path.join(outdir,'%04i.exr' % i)
        # im2 = cv2.resize(im2,(im2.shape[1]*10,im2.shape[0]*10))
        cv2.imwrite(outfn,im2)
        cv2.imwrite(outfn.replace('.exr','.png'),np.clip(im2,0,1) ** (1/2.2) * 255)
    
    fns = sorted(glob.glob('/home/mohammad/Projects/NRV/dataset/envmap-videos/pisa/*.exr'))
    out = '/home/mohammad/Projects/NRV/dataset/envmap-videos/pisa.avi'
    ims = []
    for fn in tqdm.tqdm(fns[:150]):
        im = cv2.imread(fn,-1)[:,:,::-1].copy()
        im = cv2.resize(im,(im.shape[1]*10,im.shape[0]*10))
        ims.append(im)
    ims = torch.from_numpy(np.stack(ims,0)).permute(0,3,1,2) 
    
    imageseq2avi(out,ims)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deploying command')
    parser.add_argument('--outdir',type=str, default='/home/mohammad/Projects/NRV/dataset/envmap-videos/grace/', help='Output directory')
    parser.add_argument('--envmap_fn',type=str, default='/home/mohammad/Projects/NRV/NrInit/cvgutils/tests/testimages/grace_eq.exr', help='Output directory')
    parser.add_argument('--nimgs',type=int, default=300, help='Output directory')
    parser.add_argument('--env_w',type=int, default=24, help='Output directory')
    parser.add_argument('--env_h',type=int, default=12, help='Output directory')
    args = parser.parse_args()

    rotateEnvmapSequence(args.outdir, args.envmap_fn, args.nimgs, args.env_w, args.env_h)
