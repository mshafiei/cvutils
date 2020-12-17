import cvgutils.nn.Utils as Utils
import cvgutils.Viz as viz
import cv2
import numpy as np
x = np.arange(0,100)
x0 = 10
x1 = 30
y0 = 0.1
y1 = 1e-7

interp = Utils.linearLR(x,x0,x1,y0,y1)
im = viz.plot(x,interp)
cv2.imwrite('renderout/im.png',im)