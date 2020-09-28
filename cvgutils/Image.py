import cv2
import numpy

def writePng(im, fn):
    """[Tonemap a linear image with quasi srgb (clip and gamma 1/2.2) and write in filename with png extension]

    Args:
        im ([[hxwx3 ndarray]): [linear image]
        fn (str): [filename with png extension]

    Returns:
        [hxwx3 ndarray]: [srgb image]
    """
    cv2.imwrite(fn, hdr2ldr(im))

def hdr2srgb(im):
    """[Tonemap a linear image with quasi srgb (clip and gamma 1/2.2)]

    Args:
        im ([[hxwx3 ndarray]): [linear image]

    Returns:
        [hxwx3 ndarray]: [srgb image]
    """

    return (np.clip(im,0,1) ** (1/2.2) * 255).astype(np.uint8)