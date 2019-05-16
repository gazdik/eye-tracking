#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import itertools


def plot_data(data, movement_type):
    data_fix = data[movement_type]
    data_sacc = data[np.logical_not(movement_type)]
    plt.scatter(data_sacc[:, 0], data_sacc[:, 1], color='b', s=0.01)
    plt.scatter(data_fix[:, 0], data_fix[:, 1], color='r', s=0.01)
    plt.savefig('fig.pdf')
    plt.show()

    
def plot_clusters(X, labels):
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for i in np.unique(labels):
        plt.scatter(X[labels==i, 0], X[labels==i, 1],
                    c=next(colors), s=0.01)
    plt.show()


def load_data(file_name, sid_list):
    data_mat = loadmat(file_name)
    sids = data_mat['sid'].tolist()[0]
    knowns = data_mat['known'].transpose()
    coords = data_mat['coordinates'].transpose()
    
    res = {}
    for sid, known, coord in zip(sids, knowns, coords):
        vals = [known[0], coord[0]]
        if sid[0] in sid_list:
            if sid[0] not in res:
                res[sid[0]] = []
            res[sid[0]].append(vals)
    return res


def edist(a, b):
    return np.linalg.norm(a - b, axis=1)

# Mean fixation duration
def mfd(movement_type, freq = 1000):
    durations = [] #in seconds
    for k, g in itertools.groupby(movement_type):
        g = list(g)
        if k and len(g)/freq >= 0.05: #filtering out too short fixations, probably noise
            durations.append(len(g)/ freq)
    return (np.mean(durations), len(durations)) #mean and number of fixations


def ivt(coords, threshold, frequency):
    time = 1 / frequency
    dists = edist(coords[:-1], coords[1:])
    vels = dists / time
    vels = savgol_filter(vels, 29, 2)
    print(vels[:10])
    movement_type = vels < threshold
    return movement_type
    

data = load_data('train.mat', ['s4', 's14', 's24', 's34', 's10', 's20'])
coords = data['s24'][0][1]
movement_type = ivt(coords, 3000, 1000)
print(movement_type[:10])
plot_data(coords[:-1], movement_type)
mfd(movement_type, 1000)
