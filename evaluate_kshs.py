#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:12:25 2020

@author: Alexander Sagel
"""

import numpy as np
import glob
from itertools import compress
import tqdm
from kshs_common import extract_subspace, nuclear_distance, hist_bhat_kernel, class_center


for dt in ['alpha', 'beta', 'gamma']:
    print()
    print('Computing subspaces for split', dt + '...')
    features = []
    labels = []

    for i in tqdm.tqdm(range(len(glob.glob('data/dyntex_' + dt + '/*')))):
        files = glob.glob('data/dyntex_' + dt + '/c' + str(i+1)
                          + '_*/*_st.npy')
        Vs = []
        for f in files:
            labels.append(i)
            V = np.load(f)
            features.append(extract_subspace(V, n=5, N_tilde=15))

    labels = np.asarray(labels)

    D = np.zeros((len(features), len(features)))
    print('    Performing NN Classification...')
    for i in tqdm.tqdm(range(D.shape[0])):
        for j in range(D.shape[0]):
            D[i, j] = nuclear_distance(features[i], features[j])
    D = (D+D.T)/2
    nn = np.argmin(D+2*np.eye(len(features))*np.abs(D).max(), axis=1)
    labels = np.asarray(labels)
    print('    NN Success rate:',
          np.sum(labels[nn] == np.asarray(labels))/len(features))
    class_centers = []
    for i in range(labels[-1]+1):
        class_centers.append(class_center(
            list(compress(features, labels == i)), N_tilde=15))
    print('    Performing NCC Classification...')
    D_cc = np.zeros((len(features), labels[-1]+1))
    for i in tqdm.tqdm(range(D_cc.shape[0])):
        for j in range(D_cc.shape[1]):
            if labels[i] == j:
                cc_index = labels == j
                cc_index[i] = False
                cc = class_center(list(compress(features, cc_index)),
                                  N_tilde=15)
            else:
                cc = class_centers[j]
            D_cc[i, j] = nuclear_distance(features[i], cc)
    ncc = np.argmin(D_cc, axis=1)
    print('NCC Success rate:', np.sum(labels == ncc)/len(features))
    print()
