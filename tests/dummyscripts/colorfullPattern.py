import cv2
import numpy as np

if __name__ == "__main__":
    im = (np.random.rand(5,5,3) * 255).astype(np.uint8)
    im = cv2.resize(im,(512,512),interpolation = cv2.INTER_NEAREST)
    cv2.imwrite('cvgutils/tests/testimages/5x5pattern.png',im)