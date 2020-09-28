#few basic methods for rendering with mitsuba2
#https://github.com/mitsuba-renderer/mitsuba2

import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.core import FileStream, xml, ScalarTransform4f, Bitmap, Struct
from mitsuba.render import Integrator
import cv2
import numpy as np
import torch
# import plotly.graph_objects as go
def samplePlastic(diffuse, specular, nonlinear, intior,extior):
    return {
            "type" : "roughplastic",
            "diffuse_reflectance" : {
                "type" : "rgb",
                "value" : [diffuse, diffuse, diffuse],
            },
            'nonlinear':False,
            'int_ior':intior,
            'ext_ior':extior,
            'specular_reflectance':{
                "type" : "rgb",
                "value" : [specular,specular,specular],
            }
    }


def genXML(lx, ly,material,scale):
    scene = xml.load_dict({
        "type" : "scene",
        "myintegrator" : {
            "type" : "path",
        },
        "mysensor" : {
            "type" : "perspective",
            "near_clip": 0.1,
            "far_clip": 1000.0,
            "to_world" : ScalarTransform4f.look_at(origin=[0.0, 0.001, 1],
                                                target=[0, 0, 0],
                                                up=[0, 0, 1]),
            "myfilm" : {
                "type" : "hdrfilm",
                "rfilter" : { "type" : "box"},
                "width" : 256,
                "height" : 256,
            },
            "mysampler" : {
                "type" : "independent",
                "sample_count" : 4,
            },
        },
        "myemitter" : {"type" : "point","intensity":1.0,'position':[lx * scale,ly * scale,1.1]},
        "myshape" : {
            "type" : "sphere",
            "radius": 0.2,
            "mybsdf" : material,
        }
    })
    return scene

def renderScene(scene):
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    # The rendered data is stored in the film
    film = sensor.film()

    # Write out data as high dynamic range OpenEXR file
    film.set_destination_file('./output/tmp.png')
    film.develop()
    bmp = film.bitmap(raw=True)
    # Get linear pixel values as a numpy array for further processing
    bmp_linear_rgb = bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
    image_np = np.array(bmp_linear_rgb)
    return image_np