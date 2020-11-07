import cvgutils.Image as im
import cvgutils.Linalg as lin
import numpy as np
import torch
import cv2
if __name__ == "__main__":
    probefn = 'cvgutils/tests/testimages/grace_probe.hdr'
    w = 512
    h = 256
    probeim = cv2.imread(probefn,-1)[:,:,::-1].copy()
    u = np.linspace(0,w-1,w)/w
    v = np.linspace(0,w-1,w)/h
    u, v = np.meshgrid(u, v)
    p,t = lin.uv2pt(u,v)
    x,y,z = lin.pt2xyz(p,t)
    r = 1/np.pi * np.arccos(z) / ((x**2+y**2)**0.5 + 1e-20)
    up, vp = x*r , y*r
    uv = np.stack((up,vp),axis=-1)[None,...] * 1000
    eq = torch.nn.functional.grid_sample(torch.Tensor(probeim[None,...]).permute(0,3,1,2),torch.Tensor(uv))
    im.writePng('./renderout/eq.png',eq[0].permute(1,2,0))