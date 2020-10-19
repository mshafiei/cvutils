import cvgutils.Image as im
import numpy as np
if __name__ == "__main__":
    a = np.random.rand(100,100,3)
    a = a.transpose((2,0,1))[None,...]
    im.imageseq2avi('renderout/tst.avi',a)
