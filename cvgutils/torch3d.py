import pytorch3d.renderer.cameras as torchCam
import pytorch3d.transforms.transform3d as torch3d
import numpy as np
import torch

def getCamera(w,h,near,far,fov,Origin,LookAt,Up):
    aspect = w / h
    R = torchCam.look_at_rotation(Origin,LookAt,Up)
    return torchCam.FoVPerspectiveCameras(near,far,aspect,fov,R=R,T=Origin).get_projection_transform()
        