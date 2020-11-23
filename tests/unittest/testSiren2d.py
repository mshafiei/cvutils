from cvgutils.nn.Siren import Siren
from cvgutils.nn.SirenUtils import gradient
import torch
if __name__ == "__main__":
    nn = Siren(3,3,3,1,outermost_linear=True)
    inpt = torch.ones((10,3))
    output = torch.ones((10,1))
    # inpt.requires_grad = False
    tst, coords = nn(inpt)
    loss = ((output - tst) ** 2).sum() ** 0.5
    gradient(tst,coords)