import torch
import matplotlib.pyplot as plt
n = 1000
m = 20
t = torch.linspace(-1,1,n)[None,:]
w = torch.rand(m)[:,None] * 10000
b = torch.rand(m)[:,None] * 10000
y0 = torch.rand(m)[:,None] * 100
f = -torch.relu(w * t + b) + y0
f = f.sum(-2)
f = torch.relu(f)
fig = plt.figure()
print(b)
plt.plot(t[0,:].numpy(),f.numpy())
plt.xlim(-1,1)
plt.ylim(0,1300)
plt.savefig('renderout/piecewise_linear.png')