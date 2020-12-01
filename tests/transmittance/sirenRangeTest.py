import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
# import skimage
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import wandb
import time
import tqdm
import cvgutils.Viz as viz
from cvgutils.nn.Siren import Siren
from cvgutils.nn.SirenUtils import laplace, gradient, get_mgrid


if __name__ == "__main__":
    wandb.init(project='specular_relighting',name='Transmittance 1D')
    
    visstep = 100

    img_siren = Siren(in_features=1, out_features=1, hidden_features=20, 
                    hidden_layers=2, outermost_linear=True)
    img_siren.cuda()

    total_steps = 50000 # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
    logger = viz.logger('','wandb','specular_relighting','sirenRangeTest')

    x0 = np.linspace(0,1,1000)[:,None]
    y0 = ((0.25 < x0) & (x0 < 0.75)) * 1000
    # y0 = x0 * 2
    x0 = torch.Tensor(x0).cuda()
    y0 = torch.Tensor(y0).cuda()
    batchsize = 10
    for step in tqdm.trange(total_steps):
        idx = (np.random.rand(batchsize) * x0.shape[0])
        x = x0[torch.Tensor(idx).long(),:]
        y = y0[torch.Tensor(idx).long(),:]
        model_output, coords = img_siren(x)
        loss = ((model_output - y) ** 2).mean()
        if(step % visstep == 0):    
            yout,_ = img_siren(x0)
            plt1 = viz.plot(x0.detach().cpu().numpy(),y0.detach().cpu().numpy(),ptype='plot')
            plt2 = viz.plot(x0.detach().cpu().numpy(),yout.detach().cpu().numpy(),ptype='plot')
            pltmain = np.concatenate((plt1,plt2),axis=1)
            logger.addImage(pltmain,'comparison')
            logger.addLoss(loss.detach().cpu().numpy(),'loss')

        logger.takeStep()
        optim.zero_grad()
        loss.backward()
        optim.step()