#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:01:23 2020

@author: Alexander Sagel
"""

import numpy as np
from sklearn.cluster import KMeans
global NBINS
NBINS = 20


def hist_bhat_kernel(X, Y):
    '''
    Computes the Bhattacharyya kernel matrix between to sets of Scattering
    histogram vectors

    Parameters
    ----------
    X : Numpy array,
        Scattering histogram vectors (N_1 x S x B),
        S=#of Scattering subbands, B=# of histogram bins.
    Y : Numpy array,
        Scattering histogram vectors (N_2 x S x B).

    Returns
    -------
    K : Numpy array,
        Bhattacharyya kernel matrix (N_1, N_2).

    '''
    sqrtprod = np.sqrt(X.reshape(X.shape[-3], 1, X.shape[-2], NBINS)*Y.reshape(
        1, Y.shape[-3], Y.shape[-2], NBINS)).sum(axis=-1)
    K = np.exp(np.log(sqrtprod+1e-10).sum(axis=-1))
    return K


def extract_subspace(H, n=5, N_tilde=100):
    '''
    Computes kernel subspace from a sequence of Scattering histogram vectors

    Parameters
    ----------
    H : Numpy array,
        Sequence of Scattering histogram vectors (N x S x B),
        S=#of Scattering subbands, B=# of histogram bins.
    n : Integer, optional
        Subspace dimension. The default is 5.
    N_tilde : Integer, optional
        Number of columns to be samples from H. The default is 100.

    Returns
    -------
    H_tilde: Numpy array,
        Subsampled version of H (N_tilde x S x B)
    C_tilde : Numpy array,
        Coefficient matrix (N_tilde x n).

    '''
    K = np.zeros((H.shape[0], H.shape[0]))
    for k in range(K.shape[0]):
        K[k, :] = hist_bhat_kernel(H[[k]], H)
    U, S, VT = np.linalg.svd(K)
    C = U[:, :n]*1/np.sqrt(S[:n]).reshape(1, n)
    U_tilde, S_tilde, V_tilde = np.linalg.svd(
        K[::K.shape[0]//N_tilde, ::K.shape[0]//N_tilde][:N_tilde, :N_tilde])
    C_tilde = U_tilde*1/np.sqrt(S_tilde).reshape(1, -1)
    U, S, VT = np.linalg.svd(np.dot(C_tilde.T, np.dot(K[::K.shape[0]//N_tilde,
                                                        :][:N_tilde], C)))
    C_tilde = np.dot(np.dot(C_tilde, U[:, :n]), VT)
    H_tilde = H[::K.shape[0]//N_tilde][:N_tilde]
    return (H_tilde, C_tilde)


def class_center(spaces, N_bar=200):
    '''
    Computes the Frechet mean from a set of space parameter pauirs

    Parameters
    ----------
    spaces : List,
        Sequence of parameter pairs.
    N_bar : Integer, optional
        # of Cluster centers for k-means. The default is 200.

    Returns
    -------
    H_bar, C_bar : Numpy arrays,
        Estimated Frechet mean
    '''
    kk = KMeans(N_bar)
    H = []
    Q = []
    for f in spaces:
        H.append(f[0])
        Q.append(np.eye(f[1].shape[1]))
    H = np.concatenate(H)
    H_bar = kk.fit(H.reshape(-1, H.shape[-2]*H.shape[-1])
                   ).cluster_centers_.reshape(N_bar, H.shape[-2],
                                              H.shape[-1])
    H_bar = np.where(H_bar < 0, 0, H_bar)
    U, S, _ = np.linalg.svd(hist_bhat_kernel(H_bar, H_bar))
    C_ = U/np.sqrt(S.reshape(1, -1))
    KC = []
    for i in(range(len(spaces))):
        KC.append(np.dot(hist_bhat_kernel(H_bar, spaces[i][0]),
                         spaces[i][1]))
    for it in range(10):
        TC = 0
        for i in range(len(spaces)):
            TC += np.dot(KC[i], Q[i])
        TC /= len(spaces)
        U, S, VT = np.linalg.svd(np.dot(C_, TC))
        C_bar = np.dot(np.dot(C_, U[:, :TC.shape[1]]), VT)
        for i in range(len(spaces)):
            C_barTC_i = np.dot(C_bar.T, KC[i])
            U, S, VT = np.linalg.svd(C_barTC_i)
            Q[i] = np.dot(U[:, :TC.shape[1]], VT).T
    return (H_bar, C_bar)


def nuclear_distance(xi_1, xi_2):
    '''
    Computes the Nuclear distance

    Parameters
    ----------
    xi_1 : Tuple,
        Subspace Parameter pair.
    xi_2 : Tuple,
        Subspace Parameter pair.

    '''
    K = hist_bhat_kernel(xi_1[0], xi_2[0])
    S = np.linalg.svd(np.dot(np.dot(xi_1[1].T, K), xi_2[1]), compute_uv=False)
    return 2*(xi_2[1].shape[1]-S.sum())
