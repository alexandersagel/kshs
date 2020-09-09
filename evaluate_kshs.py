#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:12:25 2020

@author: Alexander Sagel
"""

import numpy as np
import glob
import os
from sklearn.cluster import KMeans
from itertools import compress
import tqdm

global NBINS
NBINS = 20


def hist_bhat_kernel(X, Y):
    '''
    Computes the Bhattacharyya kernel matrix between to sets of Scattering
    histogram vectors

    Input
        X, Y: Numpy arrays containing Scattering histogram vectors with the
              dimensions (N_1 x S x B), (N_2 x S x B)
    Output
        K:    Bhattacharyya kernel matrix (N_1, N_2)
    '''
    sqrtprod = np.sqrt(X.reshape(X.shape[-3], 1, X.shape[-2], NBINS)*Y.reshape(
        1, Y.shape[-3], Y.shape[-2], NBINS)).sum(axis=-1)
    K = np.exp(np.log(sqrtprod+1e-10).sum(axis=-1))
    return K


def extract_klds(Y, n=5, N_tilde=100):
    K = np.zeros((Y.shape[0], Y.shape[0]))
    for k in range(K.shape[0]):
        K[k, :] = hist_bhat_kernel(Y[[k]], Y)
    U, S, VT = np.linalg.svd(K)
    R = U[:, :n]*1/np.sqrt(S[:n]).reshape(1, n)
    U_tilde, S_tilde, V_tilde = np.linalg.svd(
        K[::K.shape[0]//N_tilde, ::K.shape[0]//N_tilde][:N_tilde, :N_tilde])
    R_tilde = U_tilde*1/np.sqrt(S_tilde).reshape(1, -1)
    U, S, VT = np.linalg.svd(np.dot(R_tilde.T, np.dot(K[::K.shape[0]//N_tilde,
                                                        :][:N_tilde], R)))
    R_tilde = np.dot(np.dot(R_tilde, U[:, :n]), VT)
    return (Y[::K.shape[0]//N_tilde][:N_tilde], R_tilde)


def class_center(features, N_tilde=200):
    kk = KMeans(N_tilde)
    Y = []
    Q = []
    for f in features:
        Y.append(f[0])
        Q.append(np.eye(f[1].shape[1]))
    Y = np.concatenate(Y)
    Y_tilde = kk.fit(Y.reshape(-1, Y.shape[-2]*Y.shape[-1])
                     ).cluster_centers_.reshape(N_tilde,
                                                Y.shape[-2], Y.shape[-1])
    Y_tilde = np.where(Y_tilde < 0, 0, Y_tilde)
    U, S, _ = np.linalg.svd(hist_bhat_kernel(Y_tilde, Y_tilde))
    R_ = U/np.sqrt(S.reshape(1, -1))
    KR = []
    for i in(range(len(features))):
        KR.append(np.dot(hist_bhat_kernel(Y_tilde, features[i][0]),
                         features[i][1]))
    for it in range(10):
        TC = 0
        for i in range(len(features)):
            TC += np.dot(KR[i], Q[i])
        TC /= len(features)
        U, S, VT = np.linalg.svd(np.dot(R_, TC))
        R_tilde = np.dot(np.dot(R_, U[:, :TC.shape[1]]), VT)
        for i in range(len(features)):
            C_tildeTC_i = np.dot(R_tilde.T, KR[i])
            U, S, VT = np.linalg.svd(C_tildeTC_i)
            Q[i] = np.dot(U[:, :TC.shape[1]], VT).T
    return (Y_tilde, R_tilde)


def alignment_distance(xi_1, xi_2):
    K = hist_bhat_kernel(xi_1[0], xi_2[0])
    S = np.linalg.svd(np.dot(np.dot(xi_1[1].T, K), xi_2[1]), compute_uv=False)
    return 2*(xi_2[1].shape[1]-S.sum())


for dt in ['alpha', 'beta', 'gamma']:
    print()
    print('Processing split', dt + '...')
    features = []
    labels = []

    for i in tqdm.tqdm(range(len(glob.glob('data/dyntex_' + dt + '/*')))):
        files = glob.glob('data/dyntex_' + dt + '/c' + str(i+1)
                          + '_*/*_st.npy')
        Vs = []
        for f in files:
            labels.append(i)
            V = np.load(f)
            features.append(extract_klds(V, n=5, N_tilde=15))

    labels = np.asarray(labels)

    D = np.zeros((len(features), len(features)))
    print('    Performing NN Classification...')
    for i in tqdm.tqdm(range(D.shape[0])):
        for j in range(D.shape[0]):
            D[i, j] = alignment_distance(features[i], features[j])
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
            D_cc[i, j] = alignment_distance(features[i], cc)
    ncc = np.argmin(D_cc, axis=1)
    print('NCC Success rate:', np.sum(labels == ncc)/len(features))
    print()
