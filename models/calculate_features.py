# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from glob import 
# %%
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from glob import glob

# %%
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from glob import glob



# %%
files_2048 = glob('../features/**/2048-*.pt', recursive = True) 
files_162 = glob('../features/**/162-*.pt', recursive = True) 


# %%
num_classes = 162
for f in zip(files_2048, files_162):
    # load the feat
    feat_2048 = torch.load(f[0])
    feat_162 = torch.load(f[1])

    # size of each feat
    size_2048 = feat_2048.size()
    size_162 = feat_162.size()

    # features = torch.zeros((size_162[0], size_162[1]))
    x = nn.functional.interpolate(feat_162, size=(size_2048[2], size_2048[3]), mode='bilinear', align_corners=False)

    # get corresponding coordinates
    x = x.permute((0,2,3,1))
    features_coords = torch.zeros((size_2048[2], size_2048[3], 1))

    # find most probable class at each location, scale to 2048 dims
    for row in range(size_2048[2]):
        for col in range(size_2048[3]):
            max = torch.argmax(x[0][row][col])
            features_coords[row][col] = max

    # calculate average for each class in 2048d
    feat_2048 = feat_2048.permute((0,2,3,1)).type(torch.FloatTensor)
    all_feats = torch.zeros((num_classes, 2048)).type(torch.FloatTensor)
    for cl in range(num_classes):
        coords = np.argwhere(features_coords.numpy() == cl)
        for coord in coords:
            all_feats[cl] = all_feats[cl].add(feat_2048[0][coord[0]][coord[1]])
        all_feats[cl] = torch.div(all_feats[cl], len(coords))
        
            
            


