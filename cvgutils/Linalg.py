import numpy as np
import torch
import mathutils
def pt2xyz(p, t, r = 1):
    """[Polar to cartesian transform]

    Args:
        p ([Tensor bx1x...]): [phi \in [0,2pi]]
        t ([Tensor bx1x...]): [theta \in [0,pi]]
        r (int or Tensor bx1x..., optional): [radius > 0]. Defaults to 1.

    Returns:
        [x,y,z]: [tuple of bx1x... Tensors]
    """

    if(type(p) == torch.Tensor):
        cos = torch.cos
        sin = torch.sin
    else:
        cos = np.cos
        sin = np.sin

    x = r * sin(t) * cos(p)
    y = r * sin(t) * sin(p)
    z = r * cos(t)

    return x,y,z

def xyz2pt(x, y, z):
    """[cartesian to polar transform]

    Args:
        x ([ndarray or tensor]): [description]
        y ([ndarray or tensor]): [description]
        z ([ndarray or tensor]): [description]

    Returns:
        [tuple]: [phi \in [0,2\pi] and theta \in [0, \pi]]
    """
    assert(type(x) == type(y),'type mismatch')
    assert(type(y) == type(z),'type mismatch')

    if(type(x) == torch.Tensor):
        acos = torch.acos
        atan2 = torch.atan2
        atan = torch.atan
    else:
        acos = np.arccos
        atan2 = np.arctan2
        atan = np.arctan

    r = (x**2+y**2+z**2) ** 0.5
    p = atan2(y,x)
    p[p<0] = p[p<0] + 2*np.pi
    t = acos(z/(r+1e-20))

    return p, t

def pt2uv(p,t):
    """[polar to uniform]

    Args:
        p ([ndarray or tensor]): [description]
        t ([ndarray or tensor]): [description]
    Returns:
        [tuple]: [u \in [0,1] and v \in [0,1]]
    """
    return p/(np.pi * 2),t/np.pi

def uv2pt(u,v):
    """[Converts polar to cartesian]

    Args:
        u ([Tensor or ndarray]): [Azimuth \in [0,1]]
        v ([Tensor or ndarray]): [Elevation \in [0,1]]

    Returns:
        [tuple of Tensor or ndarray]: [\phi \in [0-2 \pi], theta \in [0,\pi] ]
    """
    t = v * np.pi
    p = u * 2 * np.pi
    return p,t

def uv2ptUniform(u,v):
    """[Converts polar to cartesian uniformly on a sphere]

    Args:
        u ([Tensor or ndarray]): [Azimuth \in [0,1]]
        v ([Tensor or ndarray]): [Elevation \in [0,1]]

    Returns:
        [tuple of Tensor or ndarray]: [\phi \in [0-2 \pi], theta \in [0,\pi] ]
    """
    if(type(v) == np.ndarray):
        acos = np.arccos
    elif(type(v) == torch.Tensor):
        acos = torch.acos

    t = acos(1- 2 * v)
    p = u * 2 * np.pi
    return p,t

def naiveSampleOnSphere(u,v):
    p,t = uv2pt(u,v)
    return pt2xyz(p,t)

def uniformSampleOnSphere(u,v):
    p,t = uv2pt(u,v)
    return pt2xyz(p,t)

def vectorNormalize(v):
    """[Normalize a vector as v/||v||]

    Args:
        v ([Tensor b,1,...]): [Input vector]

    Returns:
        [Tensor b,1,...]: [Normalized vector]
    """
    return v / ((v**2).sum(dim=1) ** 0.5 + 1e-20)

def relerr(a,b):
    return (((a-b) ** 2 / b **2) ** 0.5).sum()

def normal2frame(n):
    """[Creates a nxn local frame given a normal vector]

    :param n: [nd Normal vector]
    :type n: [Array]
    :return: [Local frame]
    :rtype: [nxn ndarray]
    """
    r = np.random.rand(len(n)); r / np.linalg.norm(r)
    y = np.cross(n,r)
    x = np.cross(r,y)
    return np.stack((x,y,n),axis=0)
    
def lookAt(Origin, LookAt, Up):

    """[Creates camera matrix in right hand coordinate. Implementation of https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function]

    Args:
        Origin ([Tensor b,3,...]): [Camera Origin in world space]
        LookAt ([Tensor b,3,...]): [Camera lookat position in world space]
        Up ([Tensor b,3,...]): [Camera Up direction in world space]

    Returns:
        [Tensor b,4,4,...]: [Camera extrinsic matrix]
    """
    Forward = LookAt - Origin
    Forward = vectorNormalize(Forward)
    Up = vectorNormalize(Up)

    Right = torch.cross(Up, Forward)
    Right = vectorNormalize(Right)
    newUp = torch.cross(Forward, Right)
    newUp = vectorNormalize(newUp)
    m = torch.stack((Right,newUp,Forward),dim=1)
    newOrigin = torch.einsum('bmn...,bn...->bm...', m, Origin)
    m = torch.cat((m,newOrigin.unsqueeze(2)),dim=2)
    m = torch.cat((m,torch.zeros_like(m)[:,0:1,:,...]),dim=1)
    m[:,3,3,...] = 1
    return m



def perspective(fov,near,far):
    """[Generate 4x4 perspective transform matrix. Re-implementation of Mitsuba]

    Args:
        near ([float]): [near plane]
        far ([float]): [far plane]
        fov ([float]): [Field of view in degrees]

    Returns:
        [ndarray]: [4x4 projection matrix]
    """
    #mitsuba
    recip = 1.0 / (far - near)
    cot = 1.0 / np.tan(np.deg2rad(fov * 0.5))

    perspective = torch.diagflat(torch.Tensor([cot,cot,far*recip,0.0]))
    perspective[2,3] = -near * far * recip
    perspective[3,2] = 1.0

    return perspective

def perspective_projection(fov,near,far,filmSize=np.array([1,1]),cropSize=np.array([1,1]),cropOffset=np.array([0,0])):
    """[Reimplementation of Mitsuba perspective_projection function]

    Args:
        fov ([float]): [Field of view in degrees]
        near ([float]): [Near plane]
        far ([float]): [Far plane]
        filmSize ([1x2 ndarray], optional): [Film size]. Defaults to np.array([1,1]).
        cropSize ([1x2 ndarray], optional): [crop size]. Defaults to np.array([1,1]).
        cropOffset ([1x2 ndarray], optional): [Crop offset]. Defaults to np.array([0,0]).

    Returns:
        [4x4 tensor]: [Perspective camera projection matrix]
    """

    aspect = filmSize[0] / filmSize[1]
    rel_offset = cropOffset / filmSize
    rel_size = cropSize / filmSize
    p = perspective(fov,near,far)

    translate = torch.eye(4)
    translate[:3,-1] = torch.Tensor([-1.0, -1.0 / aspect, 0.0])
    scale = torch.diagflat(torch.Tensor([-0.5,-0.5*aspect,1.0,1.0]))
    translateCrop = torch.eye(4)
    translateCrop[:3,-1] = torch.Tensor([-rel_offset[0],-rel_offset[1],0.0])
    scaleCrop = torch.diagflat(torch.Tensor([1/rel_size[0],1/rel_size[1],1.0,1.0]))
    m1 = torch.mm(scaleCrop,torch.mm(translateCrop,torch.mm(scale,torch.mm(translate,p))))
    return m1

def sampleRay(h,w,far,near,fov,samples,ext):
    """[Reimplementation of Mitsuba sample_ray]

    Args:
        far ([float]): [float]
        near ([float]): [float]
        fov ([float]): [float]

    Returns:
        [4x4 array]: [float]
    """
    
    aspect = w/h
    
    camera_to_sample = perspective_projection(fov,near,far,filmSize=np.array([w,h]),cropSize=np.array([w,h]))
    sample_to_camera = torch.inverse(camera_to_sample[None,...])
    samples = np.concatenate((samples,np.zeros_like(samples[:,0:1,...]),np.ones_like(samples[:,0:1,...])),axis=1)
    
    d = torch.einsum('abc,acd->abd',sample_to_camera,torch.Tensor(samples))
    d = d[:,:3,...] / d[:,3,...]
    d = vectorNormalize(d)
    d = torch.einsum('abc,ac...->ab...',ext[:,:3,:3,...].transpose(2,1),d[:,:3,...])
    
    return d

def u1tou2(u1,u2):
    """[Compute shortest rotation between two vectors]

    Args:
        u1 ([ndarray]): [1st vector]
        u2 ([ndarray]): [2nd vector]

    Returns:
        [Quaternion]: [Shortest rotation]
    """
    u1 = u1 / np.sum(u1 ** 2) ** 0.5
    u2 = u2 / np.sum(u2 ** 2) ** 0.5
    a = np.cross(u1,u2)
    t = np.arccos(np.sum(u1 * u2))
    q = mathutils.Quaternion(a,t)
    return q

#From pytorch3d
#https://github.com/facebookresearch/pytorch3d/blob/7e986cfba8e8e09fbd24ffc1cbfef2914681e02c/pytorch3d/transforms/rotation_conversions.py
def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.
    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.
    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

#From pytorch3d
#https://github.com/facebookresearch/pytorch3d/blob/7e986cfba8e8e09fbd24ffc1cbfef2914681e02c/pytorch3d/transforms/rotation_conversions.py
def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

#From pytorch3d
#https://github.com/facebookresearch/pytorch3d/blob/7e986cfba8e8e09fbd24ffc1cbfef2914681e02c/pytorch3d/transforms/rotation_conversions.py
def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

#implemented from http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
def raySphereIntersect(A,B,C,r):
    """[Find two points per valid ray]

    Args:
        A ([Tensor or ndarray]): [ray origin]
        B ([Tensor or ndarray]): [ray direction]
        C ([Tensor or ndarray]): [center of sphere]
        r ([Tensor or ndarray]): [radius of sphere]

    Returns:
        [tuple]: [t0,t1,valid]
    """
    # A = torch.Tensor([-1,-1,-1]).to(device='cuda:0') * 3
    # B = torch.Tensor([1,1,1]).to(device='cuda:0')
    # # A = torch.nn.functional.normalize(A.unsqueeze(0))
    # B = torch.nn.functional.normalize(B.unsqueeze(0))

    # a = (B ** 2).sum(-1,keepdim=True)
    # b = 2 * B.sum(-1,keepdim=True)
    # c = (A ** 2).sum(-1,keepdim=True) - ((r - C) ** 2).sum(-1,keepdim=True)
    # d = (b**2 - 4*a*c)
    # valid = d > 0
    # t0 = (-b - (d*valid) ** 0.5) / (2 * a)
    # t1 = (-b + (d*valid) ** 0.5) / (2 * a)

    a = (B ** 2).sum(-1,keepdim=True)
    b = 2 * (B * (A - C)).sum(-1,keepdim=True)
    c = ((A - C) ** 2).sum(-1,keepdim=True) - r ** 2
    d = (b**2 - 4*a*c)
    valid = d > 0
    t0 = (-b - (d*valid) ** 0.5) / (2 * a)
    t1 = (-b + (d*valid) ** 0.5) / (2 * a)

    return t0,t1,valid

#according to https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/#:~:text=In%20order%20to%20sample%20the,several%20kinds%20of%20NDF%20around.
def ggxpdf(theta, alpha):
    if(type(alpha) == np.ndarray):
        cos = np.cos
        sin = np.sin
    else:
        cos = torch.cos
        sin = torch.sin

    return (2 * alpha**2 * cos(theta) * sin(theta)) / ((alpha ** 2 - 1) * cos(theta) ** 2 + 1) ** 2

#according to https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/#:~:text=In%20order%20to%20sample%20the,several%20kinds%20of%20NDF%20around.
def ggxSample(u,alpha):
    """
    [Samples NDF given random number u]
    [u] random number
    [alpha]

    [Return] phi and theta
    """
    if(type(u) == np.ndarray):
        atan = np.arctan
        random = np.random.rand
        stack = lambda x : np.stack(x,axis=-1)
    else:
        atan = torch.atan
        random = torch.rand_like
        stack = lambda x : torch.stack(x,dim=-1)
    
    p,t = random(u) * np.pi * 2, atan(alpha * (u / (1-u)) ** 0.5)
    n = stack(pt2xyz(p,t))
    return n, ggxpdf(t,alpha)

# def BalanceHeuristic(pdfs):
#     balanced = []
#     denom = 
def sampleEnvmap(xpdf,ypdf,u,v,envmap):
    u = sampleInvCDF1D(xpdf,u).float() / envmap.shape[1]
    v = sampleInvCDF1D(ypdf,v).float() / envmap.shape[0]
    uv = torch.stack((u,v),dim=-1).unsqueeze(0) * 2 - 1
    light_intensity = torch.nn.functional.grid_sample(envmap.unsqueeze(0).permute(0,3,1,2),uv).permute(0,2,3,1)
    p, t = uv2pt(u,v)
    x,y,z = pt2xyz(p,t)
    light_dir = torch.stack((-x,y,z),dim=-1).unsqueeze(0)
    return light_intensity, light_dir#, xpdf * ypdf

    #   /* Based on "Building an Orthonormal Basis, Revisited" by
    #    Tom Duff, James Burgess, Per Christensen,
    #    Christophe Hery, Andrew Kensler, Max Liani,
    #    and Ryusuke Villemin (JCGT Vol 6, No 1, 2017) */  
    # implemented from
    #https://github.com/mitsuba-renderer/mitsuba2/blob/93baa3c548c43d4a84a04f0349b649759a783a69/include/mitsuba/core/vector.h
def shFrame(n):
    if(type(n) == np.ndarray):
        sign = np.sign
        inv = np.linalg.inv
        stack = lambda x: np.stack(x,axis=-1)
    else:
        sign = torch.sign
        inv = torch.inverse
        stack = lambda x: torch.stack(x,dim=-1)

    sz = sign(n[...,2])
    sz[sz == 0] = 1
    a = -1 / (sz + n[...,2])
    b = n[...,0] * n[...,1] * a

    x = stack((sz * n[...,0] ** 2 * a + 1.0, sz * b, -sz * n[...,0]))
    y = stack((b, sz + n[...,1] ** 2 * a, -n[...,1]))
    sh = stack((x,y,n))
    invsh = inv(sh)
    return sh, invsh


def sampleInvCDF1D(pdf,u,dim=0):
    """[Sample on a 1d pdf by inverse CDF method]

    Args:
        pdf ([ndarray]): [pdf]
        u ([ndarray]): [uniform samples]
        dim ([ndarray]): [dimension on which cdf is defined]
    """
    if(type(pdf) == np.ndarray):
        cumsum = np.cumsum
        linspace = np.linspace
        argmax = lambda x: np.argmax(x,axis=0)
        toint = lambda x: x.astype(np.int)
    else:
        cumsum = lambda x: torch.cumsum(x,dim=0)
        linspace = lambda x,y,z: torch.linspace(x,y,z,device=pdf.device)
        argmax = lambda x: torch.Tensor(np.argmax(x.cpu().numpy(),axis=0)).to(device=x.device).long()
        toint = lambda x: x.long()

    res = pdf.shape[dim]
    cdf = cumsum(pdf)
    idx = linspace(0,1,res)
    diff = (cdf[:,None] - idx[None,:]) >= 0
    I = argmax(diff * 1)
    I[-1] = res - 1
    idx2 = u * (res-1)
    return I[toint(idx2)]

