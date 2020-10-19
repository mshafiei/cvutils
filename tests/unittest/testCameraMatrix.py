#This script renders input data for Deep Reflectance Volume
import cvgutils.Mitsuba2XML as mts
import cvgutils.Image as im
import cvgutils.Linalg as lin
import cvgutils.Dir as dr
import cvgutils.torch3d as t3d
import cv2
import numpy as np
import torch
import os

import pytorch3d.renderer.cameras as torchCam
import pytorch3d.transforms.transform3d as torch3d

if __name__ == "__main__":
    
    #TODO: create math module
    #TODO: create LatLong related modules

    #directories
    outdir = 'DRV'
    outfmtTorch = 'torch-%04d-%04d.png'
    outfmt = '%04d-%04d.png'
    texturefn = 'cvgutils/tests/testimages/wood_texture.jpg'
    objfn = 'cvgutils/tests/testobjs/z.obj'
    dr.createIfNExist(outdir)

    #scene params
    intior = 2.0
    extior = 1.000277
    k = 0
    alpha = 0.0
    center = [0,0,0]
    nsamplesSphere = 1000000
    nsamples = 15
    diffuseReflectance = [1.0,1.0,1.0]
    specularReflectance = [1.0,1.0,1.0]
    # diffuseReflectance = texturefn
    dxs = torch.Tensor([0.0,1.0,0.0]) * 0.2
    
    #camera param
    h,w = (256,256)
    camx, camy, camz = (-0.0,0.0,3.0)
    atx,aty,atz = (1.0,0.0,0.0)
    camOrig = torch.Tensor([[camx,camy,camz]])
    camOrigt = torch.Tensor([[camy,camx,-camz]])
    camLookAt = torch.Tensor([[atx,aty,atz]])
    camLookAtt = torch.Tensor([[aty,atx,-atz]])
    camUp = torch.Tensor([[0.0,1.0,0.0000]])
    near = 0.01
    far = 1000.0
    fov = 60.0

    ntheta = 2
    nphi = 15
    r = 1.5
    worldcoord = lin.lookAt(torch.Tensor([[0,0,0]]),torch.Tensor([[1,0,0]]),torch.Tensor([[0,0,1]]))

    material = mts.diffuse(diffuseReflectance)
    
    #sample a random position
    #sample a random small radius
    #sample random camera parameters
    #render a sphere mask on a random positon by mitsuba
    #render a sphere mask yourself
    #compare
    

    #sample on sphere
    radius = 1#np.random.rand(1)[0]
    samples = torch.rand((1,2,nsamplesSphere))
    # sphereOrig = torch.rand((1,1,3))
    cx,cy,cz = (0,0,0)
    sphereOrig = torch.Tensor([[[cx,cy,cz]]])
    sphereOrigt = torch.Tensor([[[cy,cx,-cz]]])
    x,y,z = lin.sampleUniformOnSphere(samples[:,0,:],samples[:,1,:])
    xyz = torch.stack((x,y,z),dim=2) * radius + sphereOrigt

    #rasterization by pytorch3d
    aspect = w / h
    # R,T = torchCam.look_at_view_transform(dist=2.7, elev=0, azim=0)
    R = torchCam.look_at_rotation(camOrig,camLookAtt,camUp)
    T = torch.einsum('abc,ac->ab',R,camOrig)
    cam = torchCam.FoVPerspectiveCameras(near,far,aspect,fov,R=R,T=T)
    
    uv = cam.transform_points_screen(xyz,torch.Tensor([[h,w]]))
    uv = torch.clamp(uv,0,h-1)
    img = torch.zeros((h,w,3))
    img[uv[0,:,0].long(),uv[0,:,1].long(),:] = 1
    
    outfn = os.path.join(outdir,outfmtTorch % (0,0))
    im.writePng(outfn,img)

    #render by mitsuba
    light = mts.pointlight(camOrig)
    camera = mts.camera(camOrig,camLookAt,camUp,fov,nsamples=nsamples)
    shape = mts.sphere(sphereOrig[0], radius, material)

    scene = mts.generateScene(shape,light,camera)
    img = mts.renderScene(scene) > 0
    outfn = os.path.join(outdir,outfmt % (0,0))
    im.writePng(outfn,img)