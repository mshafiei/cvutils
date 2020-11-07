import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import wandb
import time
import tqdm

from Siren import Siren
from SirenUtils import laplace, gradient, get_mgrid

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())        
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels


if __name__ == "__main__":
    wandb.init(project='specular_relighting')
    cameraman = ImageFitting(256)
    dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=2, out_features=1, hidden_features=256, 
                    hidden_layers=3, outermost_linear=True)
    img_siren.cuda()

    total_steps = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in tqdm.trange(total_steps):
        model_output, coords = img_siren(model_input)    
        loss = ((model_output - ground_truth)**2).mean()
        wandb.log({'out':[wandb.Image(model_output.view(256,256))], 'in':[wandb.Image(ground_truth.view(256,256))]},step=step)
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            # fig, axes = plt.subplots(1,3, figsize=(18,6))
            # axes[0].imshow(model_output.cpu().view(256,256).detach().numpy())
            # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256,256).detach().numpy())
            # axes[2].imshow(img_laplacian.cpu().view(256,256).detach().numpy())
            # plt.show()

        optim.zero_grad()
        loss.backward()
        optim.step()