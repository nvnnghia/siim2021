import numpy as  np
import matplotlib.pyplot as plt
import torch
import timm
from torch.optim import Optimizer

def sigmoid_rampup(current, rampup_length=15):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return 0.9*float(np.exp(-5.0 * phase * phase))


total_steps = 15
lrs = []
steps = []
for i in range(total_steps):
    # lr = optimizer.param_groups[0]['lr']
    # scheduler.step()
    lr = sigmoid_rampup(i)
    lr = 10*(1-lr)
    steps.append(i)
    lrs.append(lr)

print(lr)
fig, ax = plt.subplots()
ax.plot(steps, lrs)
ax.set(xlabel='step', ylabel='ratio',
       title='warmup scheduler')
ax.grid()
fig.savefig("test.png")