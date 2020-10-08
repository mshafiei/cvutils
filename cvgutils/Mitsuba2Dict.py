#few basic methods for rendering with mitsuba2
#https://github.com/mitsuba-renderer/mitsuba2

import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.core import FileStream, xml, ScalarTransform4f, Bitmap, Struct
from mitsuba.render import Integrator
import cv2
import numpy as np
import torch

# bsdfs
def plastic(diffuse, specular, nonlinear, intior,extior):
    """[Plastic material dict]

    Args:
        diffuse ([list]): [rgb values]
        specular ([list]): [rgb values]
        nonlinear ([bool]): [description]
        intior ([float]): [description]
        extior ([list]): [description]
        
    Returns:
        [dict]: [material dict]
    """
    return {
            "type" : "roughplastic",
            "diffuse_reflectance" : {
                "type" : "rgb",
                "value" : diffuse,
            },
            'nonlinear':False,
            'int_ior':intior,
            'ext_ior':extior,
            'specular_reflectance':{
                "type" : "rgb",
                "value" : specular,
            }
    }

def diffuse(diffuse):
    if(type(diffuse) == list):
        difftype = 'rgb'
    elif(type(diffuse) == str):
        difftype = 'texture'
    else:
        raise Exception("wrong material argument")
    return {
            "type" : "diffuse",
            "reflectance" : {
                "type" : difftype,
                "value" : diffuse,
            }
    }

#shapes
def sphere(center, radius,material):
    return {
        "type":"sphere",
        "center":center,
        "radius":float(radius),
        "mybsdf":material
    }

#lights
def envmap(fn):
    return {
        "type":"envmap",
        "filename":fn
    }

#camera
def camera(origin,lookat,up,fov,w=256,h=256,nsamples=4):
    transform = ScalarTransform4f.look_at(origin=origin,
                                                target=lookat,
                                                up=up)
    
    return {
            "type" : "perspective",
            "near_clip": 0.1,
            "far_clip": 1000.0,
            "to_world" : transform,
            "fov":fov,
            "myfilm" : {
                "type" : "hdrfilm",
                "rfilter" : { "type" : "box"},
                "width" : w,
                "height" : h,
            },
            "mysampler" : {
                "type" : "independent",
                "sample_count" : nsamples,
            },
        }

    


def generateScene(shape,light,camera):
    scene = xml.load_dict({
        "type" : "scene",
        "myintegrator" : {
            "type" : "path",
        },
        "mysensor" : camera,
        "myemitter" : light,
        "myshape" : shape
    })
    return scene

def renderScene(scene):
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    # The rendered data is stored in the film
    film = sensor.film()

    # Write out data as high dynamic range OpenEXR file
    film.set_destination_file('tmp.png')
    film.develop()
    bmp = film.bitmap(raw=True)
    # Get linear pixel values as a numpy array for further processing
    bmp_linear_rgb = bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
    image_np = np.array(bmp_linear_rgb)
    return image_np