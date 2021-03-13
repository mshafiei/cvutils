from cvgutils.Image import imageseq2avi, loadImageSeq
import glob
import numpy as np
import h5py
import argparse
import os 

#1. figure from envmap
#2. figure from point light
#3. error for point light
#4. error for envmap
#5. compute ratio plot

#Creates subfigure given hierarchical image directories
#as well as errors
#input hierarchy: scene/method/filename
#Output latex file: scene x method grid of images
#Output image dir: scene_method.png

parser = argparse.ArgumentParser(description='Deploying command')
parser.add_argument('--outdir',type=str, default='../ICCVtex/', help='Directory to store latex file and images')
parser.add_argument('--indir',type=str, default='../ICCVexpsCluster/', help='Input directory containing the images')
parser.add_argument('--scenes',type=str, default='buddha2,buddha,globe,zebra', help='Comma separated scene names')
parser.add_argument('--frameids',type=str, default='127,127,110', help='frameids comma separated')
parser.add_argument('--label',type=str, default='methods', help='Common label')
parser.add_argument('--width',type=float, default=0.3, help='Width of each cell')
args = parser.parse_args()

if(not os.path.exists(args.outdir)):
    os.makedirs(args.outdir)

frameids = [int(i) for i in args.frameids.split(',')]

#make sure all images in the matrix exist
scene_list_dir = [l for l in os.listdir(args.indir) if '_scene' in l]
scenedict = {}
dr_frameids = {}
num = -1
for i, scene in enumerate(scene_list_dir):
    scene_dir = os.path.join(args.indir,scene)
    method_list_dir = [l for l in os.listdir(scene_dir) if '_method' in l]
    #same number of method per scene
    assert num == -1 or num == len(method_list_dir)
    num = len(method_list_dir)
    for method in method_list_dir:
        key = '%s_%s'%(scene.split('_scene')[0], method.split('_method')[0])
        scenedict[key] = os.path.join(scene_dir, method)
        dr_frameids[key] = frameids[i]

local_fn = {}
runtimeDict = {}
#-----------copy images to the tex directory---------------
for key, val in zip(scenedict.keys(),scenedict.values()):
    fns = [l for l in os.listdir(os.path.join(val,'render-video-latest-1')) if '.png' in l]
    fns = sorted(fns)
    idx = dr_frameids[key]
    name = [l.split('_method')[0] for l in val.split('/') if 'method' in l]
    #find max runtime sample counts
    runtimefn = os.path.join(val,name[0],'runtime.txt')
    with open(runtimefn,'r') as fd:
        ls = fd.readlines()
    runtimes = [float(l.split(' ')[0]) for l in ls if (('start' not in l) and ('overall' not in l) and ('end' not in l))]

    src = os.path.join(val,'render-video-latest-1',fns[idx])
    dst = os.path.join(args.outdir,key + '.png')
    cmd = 'cp %s %s' % (src, dst)
    local_fn[key] = dst
    runtimeDict[key] = np.array(runtimes).mean()
    print(cmd)
    os.system(cmd)


#create tex
cmd =  """\\begin{figure}\n"""
cmd +=  """      \centering\n"""

for i,(key, fn) in enumerate(zip(local_fn.keys(),local_fn.values())):
    txt = """
    \\begin{subfigure}[b]{%f\\textwidth}
            \centering
            \includegraphics[width=\\textwidth]{%s}
            \label{fig:%s}
        \end{subfigure}""" % (args.width, fn, args.label + '_' + key)
    if((i+1) % len(method_list_dir) == 0):
        txt += """
        \hfill""" 

    cmd += txt
cmd += """    \caption{Three simple graphs}\n"""
cmd += """    \label{fig:%s}\n""" % args.label
cmd += """\end{figure}\n"""
print(cmd)
with open(args.outdir + 'figure.tex','w') as fd:
    fd.write(cmd)

zipcmd = 'tar cvzf %s %s' %(args.outdir +'/zip.tgz', args.outdir)
print(zipcmd)
os.system(zipcmd)
import json
runtimeJson = json.dumps(runtimeDict,indent=True)
with open(os.path.join(args.outdir,'runtime.txt'),'w') as fd:
    fd.write(runtimeJson)
