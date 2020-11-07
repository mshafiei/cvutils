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

from cvgutils.nn.Siren import Siren
from cvgutils.nn.SirenUtils import laplace, gradient, get_mgrid
class oneDFitting(Dataset):
    def __init__(self,sidelength, y,dy, x0=-1, x1=1):
        super().__init__()
        
        self.y = y[None,].view(-1, 1)
        self.dy = dy[None,].view(-1, 1)
        self.coords = get_mgrid(sidelength, 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.y, self.dy


if __name__ == "__main__":
    wandb.init(project='specular_relighting',name='Transmittance 1D')
    sidelength = 256
    x = torch.linspace(-1,1,sidelength)
    a = -0.5
    b = 0.5
    visstep = 100
    sigma = lambda x, a, b : (a <= x) & (x <= b)
    # sigma = lambda x, a, b : np.sin(np.pi * np.maximum(0,np.minimum(x-a,b-x)))
    def tau(x,a,b,sigma):

        res = torch.exp(-(sigma(b,a,b) * b - sigma(x,a,b) * x))
        res[x < a] = torch.exp(-(sigma(torch.Tensor([b]),a,b) * b - sigma(torch.Tensor([a]),a,b) * a))
        res[x > b] = 1
        return res
    sigmaevl = sigma(x,a,b)
    ds = oneDFitting(256,y=tau(x,a,b,sigma),dy=sigmaevl)
    
    dataloader = DataLoader(ds, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=1, out_features=1, hidden_features=20, 
                    hidden_layers=2, outermost_linear=True)
    img_siren.cuda()

    total_steps = 50000 # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth, gt_dy = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in tqdm.trange(total_steps):
        model_output, coords = img_siren(model_input)    
        loss = ((model_output - ground_truth)**2).mean()
        grad = gradient(model_output,coords)
        sigma_pred = grad / (model_output + 1e-7)
        with torch.no_grad():
            # data = [[x,y] for (x,y) in zip(model_input.cpu().numpy(),model_output.cpu().numpy())]
            def wandbPlot(x,y,label,step):
                plt.figure()
                plt.plot(x,y)
                wandb.log({label:wandb.Image(plt)},step=step)
                plt.close()

            if(step % visstep == 0):
                wandbPlot(coords.cpu().numpy().reshape(-1),model_output.cpu().numpy().reshape(-1),'Transmittance Prediction',step)
                wandbPlot(model_input.cpu().numpy().reshape(-1),ground_truth.cpu().numpy().reshape(-1),'Transmittance Ground Truth',step)
                wandbPlot(coords.cpu().numpy().reshape(-1),sigma_pred.cpu().numpy().reshape(-1),'Opacity Prediction',step)
                wandbPlot(coords.cpu().numpy().reshape(-1),sigmaevl.cpu().numpy().reshape(-1),'Opacity Ground Truth',step)
            
            
            opacity_err = ((sigma_pred.cpu() - sigmaevl*1.) ** 2).mean() ** 0.5
            transmittance_err = ((model_output - ground_truth)**2).mean() ** 0.5
            wandb.log({'opacity_err':opacity_err},step=step)
            wandb.log({'transmittance_err':transmittance_err},step=step)
            # table = wandb.Table(data=data[0], columns = ["x", "f(x)"])

        # if not step % steps_til_summary:
        #     print("Step %d, Total loss %0.6f" % (step, loss))
        #     img_grad = gradient(model_output, coords)
        #     img_laplacian = laplace(model_output, coords)

            # fig, axes = plt.subplots(1,3, figsize=(18,6))
            # axes[0].imshow(model_output.cpu().view(256,256).detach().numpy())
            # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256,256).detach().numpy())
            # axes[2].imshow(img_laplacian.cpu().view(256,256).detach().numpy())
            # plt.show()

        optim.zero_grad()
        loss.backward()
        optim.step()