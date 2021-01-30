from scipy.stats import norm
import matplotlib.pyplot as plt
import cvgutils.Linalg as lin
import cv2
import numpy as np
import torch
fn = '/home/mohammad/Projects/NRV/NrArtFree/cvgutils/tests/testimages/random.exr'
imorig = cv2.imread(fn,-1)
im = imorig.mean(axis=-1)
xpdf = im.sum(axis=0)
ypdf = im.sum(axis=1)
ypdf /= ypdf.sum()
xpdf /= xpdf.sum()
n = 1000
x = lin.sampleInvCDF1D(torch.Tensor(xpdf),torch.rand(n))
y = lin.sampleInvCDF1D(torch.Tensor(ypdf),torch.rand(n))

# y,x = np.meshgrid(ysamples,xsamples)
imorig[y,x,0] = 0
imorig[y,x,1] = 0
imorig[y,x,2] = 1
cv2.imwrite('renderout/tst.png',(np.clip(imorig,0,1) ** (1/2.2) * 255).astype(np.uint8))


# fig, ax = plt.subplots(1, 1)
# mean, var, skew, kurt = norm.stats(moments='mvsk')
# x = np.linspace(norm.ppf(0.01),
#                 norm.ppf(0.99), 100)
# ax.plot(x, norm.pdf(x),
#        'r-', lw=5, alpha=0.6, label='norm pdf')

# pdf = norm.pdf(x)
# # u = np.random.rand(10)
# a = 100
# b = 50
# n = 4
# # u = np.array([1/n] * n)
# u = np.linspace(0,1,100)
# sampled = sampleInvCDF1D(pdf/pdf.sum(),u)
# ax.scatter(x[sampled],[1] * len(sampled))
# plt.show()