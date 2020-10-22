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
def plastic(diffuse, specular, intior,extior):
    xmlstr = """<bsdf type="roughplastic">
        <string name="distribution" value="beckmann"/>
        <float name="int_ior" value="%f"/>
        <float name="ext_ior" value="%f"/>
        <rgb name="diffuse_reflectance" value="%f,%f,%f"/>
        <rgb name="specular_reflectance" value="%f,%f,%f"/>
    </bsdf>""" % (intior,extior,diffuse[0],diffuse[1],diffuse[2],specular[0],specular[1],specular[2])
    return xmlstr

def conductor(specular, eta, k, alpha):
    xmlstr = """<bsdf type="roughconductor">
        <string name="distribution" value="beckmann"/>
        <float name="eta" value="%f"/>
        <float name="k" value="%f"/>
        <float name="alpha" value="%f"/>
        <rgb name="specular_reflectance" value="%f,%f,%f"/>
    </bsdf>""" % (eta,k,alpha,specular[0],specular[1],specular[2])
    return xmlstr



def diffuse(diffuse):
    if(type(diffuse) == list):
        exp = """
            <rgb name="reflectance" value="%f, %f, %f"/>
        """%(diffuse[0],diffuse[1],diffuse[2])
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
    if(type(center) == torch.Tensor):
        center = center[0]
    transform = """<transform version="2.0.0" name="to_world">
            <scale value="%f"/>
            <translate x="%f" y="%f" z="%f"/>
        </transform>"""%(radius,center[0],center[1],center[2])
    xmlstr =  """<shape version="2.0.0" type="sphere">
        %s
        %s
    </shape>""" % (transform, material)
    return xmlstr

def obj(center, scale,material,fn):

    transform = """<transform version="2.0.0" name="to_world">
            <scale value="%f"/>
            <translate x="%f" y="%f" z="%f"/>
        </transform>"""%(scale,center[0],center[1],center[2])
    xmlstr =  """<shape version="2.0.0" type="obj">
        %s
        %s
        <string name="filename" value="%s" />
    </shape>""" % (transform, material,fn)
    return xmlstr

#lights
def envmap(fn):
    xmlstr =  """ <emitter version="2.0.0" type="envmap">
        <string name="filename" value="%s"/>
    </emitter>"""%fn
    return xmlstr

def pointlight(p,intensity):
    if(type(p)==torch.Tensor):
        p = p[0]
    xmlstr =  """ <emitter version="2.0.0" type="point">
        <point name="position" value="%f,%f,%f"/>
        <rgb name="intensity" value="%f,%f,%f"/>

    </emitter>"""%(p[0],p[1],p[2],intensity[0],intensity[1],intensity[2])
    return xmlstr

#camera
def camera(origin,lookat,up,fov,ext=None,near=0.01,far=1000.0,w=256,h=256,nsamples=4):
    if(not(ext is None)):
        transform = ScalarTransform4f(ext.cpu().numpy()[0])
    elif(type(origin)==torch.Tensor):
        origin = origin[0]
        lookat = lookat[0]
        up = up[0]
        
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
    </sensor>""" %(transform,near,far,fov,film,sampler)
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

def renderRays(sensor,uv):
    h,w = np.array(sensor.film().size())
    ray = np.zeros((h,w,3))

    # for i in np.arange(h):
    #     for j in np.arange(w):
    for idx, (uv0) in enumerate(uv.transpose(1,0)):
        i = idx // h
        j = idx % h
        r, _ = sensor.sample_ray(0, 0, [uv0[0],uv0[1]], 0)
        ray[j,i,:] = np.array(r.d)
            
    return ray


