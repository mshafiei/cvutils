import cvgutils.Linalg as lin
import numpy as np
import torch
n = 1
d = 3
c0 = np.random.rand(n,3)
c1 = np.random.rand(n,3)
c0 = c0 / np.linalg.norm(c0,keepdims=True, axis=-1)
c1 = c1 / np.linalg.norm(c1,keepdims=True, axis=-1)
r = np.random.rand(n,1)*3+1
raydir = c1 - c0
raydir /= np.linalg.norm(raydir,keepdims=True, axis=-1)
rayorig = c0 - r * raydir
t0, t1, valid = lin.raySphereIntersect(torch.Tensor(rayorig),torch.Tensor(raydir),0,1)

c0p = rayorig + t0.numpy() * raydir
c1p = rayorig + t1.numpy() * raydir

print('c0 ', c0, ' ', c0p)
print('c1 ', c1, ' ', c1p)