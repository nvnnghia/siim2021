#!/usr/bin/env python
# coding: utf-8

# # Find Duplicates
# Modified from https://www.kaggle.com/appian/let-s-find-out-duplicate-images-with-imagehash


import glob
import itertools
import collections

from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import imagehash
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
        #lambda x: imagehash.whash(x, mode='db4'),
    ]

def get_hash(path):
    image = Image.open(path)
    imageid = path #.split('/')[-1] #.split('.')[0]
    h = np.array([f(image).hash for f in funcs]).reshape(256)
    return imageid, h

path1 = glob.glob('ricord1024/*jpg')
path2 = glob.glob('png1024/*png')
path3 = glob.glob('bimcv1024/*png')
paths = path1 + path2 + path3

aa = Parallel(n_jobs=16, backend="threading")(delayed(get_hash)(path) for path in tqdm(paths))

SOPInstanceUIDs = np.array([a[0] for a in aa])
hashes_all = np.array([a[1] for a in aa])

print(SOPInstanceUIDs.shape)

hashes_all = torch.Tensor(hashes_all.astype(int)).cuda()

print(hashes_all.shape)

sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).cpu().numpy()/256 for i in range(hashes_all.shape[0])])

indices1 = np.where(sims > 0.99)
indices2 = np.where(indices1[0] != indices1[1])
SOPInstanceUID1 = [SOPInstanceUIDs[i] for i in indices1[0][indices2]]
SOPInstanceUID2 = [SOPInstanceUIDs[i] for i in indices1[1][indices2]]
dups = {tuple(sorted([SOPInstanceUID1,SOPInstanceUID2])):True for SOPInstanceUID1, SOPInstanceUID2 in zip(SOPInstanceUID1, SOPInstanceUID2)}
print('found %d duplicates' % len(dups))

np.save('dups.npy', np.array(list(dups)))

count = 0
for a1, a2 in sorted(list(dups)):
    if a1[-4:] != a2[:-4]:
        count+=1

print(count)




