import cvgutils.Image as im
import numpy as np
import cv2
if __name__ =="__main__":
    fn = 'cvgutils/tests/testimages/multichannel.exr'
    imgs = im.readExrImage(fn,[['R','G','B','A'],['dd.y'],['nn.X','nn.Y','nn.Z']])
    cv2.imwrite('renderout/rgba.exr',imgs[0][:,:,:3])
    cv2.imwrite('renderout/depth.exr',imgs[1])
    cv2.imwrite('renderout/nn.exr',imgs[2][:,:,::-1])
    