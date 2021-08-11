import numpy as np 
import matplotlib.pyplot as plt

scores = np.load('score100.npy')
print(scores)
fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

axs.hist(scores, bins=50)

plt.savefig('aa.png')