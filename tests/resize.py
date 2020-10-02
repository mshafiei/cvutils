import cvgutils.Image as img
import numpy as np
import cv2

im = (cv2.imread('tests/testimages/highfreq.jpg',-1) / 255.0) ** (2.2)
res = img.resize(im,dx=256,dy=256)
cv2.imshow('hi',(res ** (1/2.2) * 255).astype(np.uint8))
cv2.waitKey(0)