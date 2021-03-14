import numpy as np
import matplotlib.pyplot as plt
import cvgutils.Viz as viz
import torch
import numpy as np
import cv2
t0 = 1
t1 = -1
As = np.linspace(0,50,5).astype(np.float32)
Bs = np.linspace(-1,1,5).astype(np.float32)
imc = []
xlim=[0,1]
ylim=[0,1]
for a in As:
    imr = []
    for b in Bs:
        # b = torch.sigmoid(torch.Tensor([b])) * 2 - 1
        t = torch.linspace(t0,t1,100)
        # f = torch.exp(-(a * t + b))
        # f = torch.sigmoid(a*(t-b))
        f = torch.exp(-a*(t-b))
        
        imr.append(viz.plot(t,f,xlim=xlim,ylim=ylim))
        # fig = plt.figure()
        # plt.plot(t.numpy(),f.numpy())
        # plt.savefig('renderout/exp.png')
    imc.append(np.concatenate(imr,axis=0))
im = np.concatenate(imc,axis=1)
cv2.imwrite('renderout/plots.png',im)

# for n, p in named_parameters:
#         if (p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())

# def plot_grad_flow_v2(named_parameters):
#     ave_grads = []
#     max_grads = []
#     OFF = 20.0
#     layers = []
#     for n, p in named_parameters:
#         if p.requires_grad and ("bias" not in n):
#             layers.append(n)

#             if p.grad is None:
#                 ave_grads.append(0)
#                 max_grads.append(0)
#             else:
#                 ave_grads.append(max(OFF + np.log(p.grad.abs().mean() + 1e-20), 0))
#                 max_grads.append(max(OFF + np.log(p.grad.abs().max() + 1e-20), 0))
#     plt.figure(figsize=(18, 24))
#     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
#     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
#     plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
#     plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical", fontsize='x-small')
#     plt.xlim(left=0, right=len(ave_grads))
#     plt.ylim(bottom=-1e-6, top=np.array(ave_grads).max())  # zoom in on the lower gradient regions
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.legend(
#         [Line2D([0], [0], color="c", lw=4),
#          Line2D([0], [0], color="b", lw=4),
#          Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient']
#     )

#     curr_time = time()
#     plt.savefig('../tmp/figures/grad_' + str(int(curr_time)) + '.png', dpi=300)