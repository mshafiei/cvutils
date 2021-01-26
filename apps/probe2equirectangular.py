import numpy as np
import cv2
import cvgutils.Linalg as lin
import argparse
import torch

parser = argparse.ArgumentParser(description='Deploying command')
parser.add_argument('--probe_fn',type=str, default='/home/mohammad/Projects/NRV/NrArtFree/cvgutils/tests/testimages/grace_probe.hdr', help='Light probe filename')
parser.add_argument('--eq_fn',type=str, default='/home/mohammad/Projects/NRV/NrArtFree/cvgutils/tests/testimages/grace_eq.exr', help='Equirectangular env map filename')
parser.add_argument('--width',type=int, default=1000, help='Equirectangular env map width')
parser.add_argument('--height',type=int, default=500, help='Equirectangular env map height')
args = parser.parse_args()

probe_im = cv2.imread(args.probe_fn, -1)
latlong_width = args.width
latlong_height = args.height


latlong_height = latlong_width // 2
latlong_pixel_num = latlong_width * latlong_height

phiv, thetav = np.meshgrid(range(latlong_width), range(latlong_height),
                           sparse=False, indexing='xy')
phiv = (phiv + 0.5) / latlong_width * 2 * np.pi
thetav = (thetav.astype(np.float32) + 0.5) / latlong_height * np.pi
coord = np.stack([-np.sin(thetav) * np.sin(phiv),
                  np.cos(thetav),
                  -np.sin(thetav) * np.cos(phiv)], axis=-1)

Dx = coord[...,0]
Dy = coord[...,1]
Dz = coord[...,2]
r = (1/np.pi) * np.arccos(Dz) / np.sqrt(Dx ** 2 + Dy ** 2)
u, v = (Dx * r, -Dy * r)
uvt = torch.Tensor(np.stack((u,v),axis=-1))[None,...]
ft = torch.Tensor(probe_im)[None,...].permute(0,3,1,2)
imeq = torch.nn.functional.grid_sample(ft,uvt)

cv2.imwrite(args.eq_fn, imeq.permute(0,2,3,1)[0].cpu().numpy())