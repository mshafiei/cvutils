import cv2
import numpy as np

if __name__ == "__main__":
    im = (np.random.rand(10,10,3) * 255).astype(np.uint8)
    im = im.astype(np.uint8)
    im = cv2.resize(im,(128,128),interpolation = cv2.INTER_NEAREST)
    # cv2.imwrite('cvgutils/tests/testimages/5x5pattern.png',im)
    cv2.imwrite('cvgutils/tests/testobjs/diffuse.png',im)
