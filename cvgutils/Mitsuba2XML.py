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
        exp = """<bsdf type="diffuse">
            <rgb name="reflectance" value="%f, %f, %f"/>
        </bsdf>"""%(diffuse[0],diffuse[1],diffuse[2])
    elif(type(diffuse) == str):
        exp = """<texture type="bitmap" name="reflectance">
                    <string name="filename" value="%s"/>
        </texture>""" % diffuse
    else:
        raise Exception("wrong material argument")
    xmlstr = """<bsdf version="2.0.0" type="diffuse">
                %s
            </bsdf>""" % exp
    return xmlstr

#shapes
def sphere(center, radius,material):
    transform = """<transform version="2.0.0" name="to_world">
            <scale value="%f"/>
            <translate x="%f" y="%f" z="%f"/>
        </transform>"""%(radius,center[0],center[1],center[2])
    xmlstr =  """<shape version="2.0.0" type="sphere">
        %s
        %s
    </shape>""" % (transform, material)
    return xmlstr

#lights
def envmap(fn):
    xmlstr =  """ <emitter version="2.0.0" type="envmap">
        <string name="filename" value="%s"/>
    </emitter>"""%fn
    return xmlstr

#camera
def camera(origin,lookat,up,fov,w=256,h=256,nsamples=4):
    transform = ScalarTransform4f.look_at(origin=origin,
                                                target=lookat,
                                                up=up)
    film = """<film type="hdrfilm">
            <integer name="width" value="%i"/>
            <integer name="height" value="%i"/>

            <rfilter type="gaussian"/>
        </film>""" %(w,h)
    sampler = """<sampler type="independent">
    <integer name="sample_count" value="%i"/>
        </sampler>""" % (nsamples)
    
    transform = """<transform version="2.0.0" name="to_world">
            <lookat origin="%f, %f, %f" target="%f, %f, %f" up="%f, %f, %f"/>
        </transform>""" % (origin[0],origin[1],origin[2],lookat[0],lookat[1],lookat[2],up[0],up[1],up[2])

    xmlstr =  """<sensor version="2.0.0" type="perspective">
        %s
        <float name="near_clip" value="%f"/>
        <float name="far_clip" value="%f"/>
        <float name="fov" value="%f"/>
        %s
        %s
    </sensor>""" %(transform,0.1,1000.0,fov,film,sampler)
    return xmlstr

    


def generateScene(shape,light,camera):
    xmlstr = """<scene version="2.0.0">
    %s
    %s
    %s
    </scene>"""%(camera,light,shape)
    scene = xml.load_string(xmlstr)
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