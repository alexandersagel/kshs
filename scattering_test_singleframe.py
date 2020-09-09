#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:12:25 2020

@author: sagel
"""

from kymatio.torch import Scattering2D
import imageio
import numpy as np
import glob
import os
import torch

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

global NBINS
NBINS = 20


def extract_hist_featires(imgs, st, normalized=None):
    imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
    stimgs = st(torch.tensor(np.float32(imgs), device='cuda')).cpu().numpy()
    nchannels = stimgs.shape[-3]
    features = np.zeros((imgs.shape[0], imgs.shape[1], nchannels, NBINS))
    for i in range(len(imgs)):
        for j in range(imgs.shape[1]):
            for l in range(nchannels):
                if normalized is not None:
                    J, L = normalized 
                    if l == 0:
                        bns = np.arange(NBINS+1)/NBINS
                        smpls = stimgs[i, j, l]
                    elif l < L*J+1:
                        bns = np.arange(NBINS+1)/NBINS
                        smpls = stimgs[i, j, l]/imgs[i,j].mean()
                    else:
                        bns = np.arange(NBINS+1)/NBINS
                        smpls = stimgs[i, j, l]/(stimgs[i, j, 2*(l-L*J-1)//(L*(J-1))+1]+1e-16)
                else:
                    if l == 0:
                        bns = np.arange(NBINS+1)/NBINS
                    else:
                        bns = np.arange(NBINS+1)/(NBINS*16)
                    smpls = stimgs[i, j, l]
                h, _ = np.histogram(smpls, bns)
                if h.sum()==0:
                    h[-1] = 1
                    print('schrei', l)
                features[i, j , l] = h/h.sum()
    return features

def readvid_gr_features(filename, st):
    '''
    this function returns a numpy array containing the video frames of the
    provided avi file converted to grayscale and scaled by the indicated
    factor.

    Output array has the dimensions (video length, height, width)
    '''
    V = []
    vid = imageio.get_reader(filename,  'ffmpeg')

    for image in vid.iter_data():
        arr = np.float32(np.asarray(image))/255.0
        grarr = 0.299*arr[:, :, 0] + 0.587*arr[:, :, 1] + 0.114*arr[:, :, 2]
        V.append(extract_hist_featires(np.expand_dims(grarr, 0), st, normalized=(4, 4)))
        break
    vid.close()
    return np.concatenate(V).reshape(len(V), -1, NBINS)


def hist_bhat_kernel(X, Y):
    sqrtprod = np.sqrt(X.reshape(X.shape[-3], 1, X.shape[-2], NBINS)*Y.reshape(1, Y.shape[-3], Y.shape[-2], NBINS)).sum(axis=-1)
    return np.exp(np.log(sqrtprod+1e-10).sum(axis=-1))


def extract_klds(Y, n=5, N_tilde=100):
    K = np.zeros((Y.shape[0], Y.shape[0]))
    for k in range(K.shape[0]):
        K[k, :] = hist_bhat_kernel(Y[[k]], Y)
    L = np.linalg.lstsq(K[:, ::K.shape[0]//N_tilde][:, :N_tilde], K)[0]
    mu = L.mean(axis=1)
    xtx = ((mu.reshape(1, -1)*mu.reshape(-1, 1))*K[::K.shape[0]//N_tilde, ::K.shape[0]//N_tilde][:N_tilde, :N_tilde]).sum()
    proj = np.eye(K.shape[0])-np.ones((K.shape[0], K.shape[0]))/K.shape[0]
    K_bar = np.dot(np.dot(proj, K), proj)
    U, S, VT = np.linalg.svd(K_bar)
    R = U[:, :n]*1/np.sqrt(S[:n]).reshape(1, n)
    U_tilde, S_tilde, V_tilde = np.linalg.svd(K[::K.shape[0]//N_tilde, ::K.shape[0]//N_tilde][:N_tilde, :N_tilde])
    R_tilde = U_tilde*1/np.sqrt(S_tilde).reshape(1, -1)
    U, S, VT = np.linalg.svd(np.dot(R_tilde.T, np.dot(K[::K.shape[0]//N_tilde, :][:N_tilde], R)))
    R_tilde = np.dot(np.dot(R_tilde, U[:, :n]), VT)
    return (Y[::K.shape[0]//N_tilde][:N_tilde], mu, xtx, R_tilde)



def alignment_distance(xi_1, xi_2, lambda_mu=1):
    xtx = xi_1[2]
    yty = xi_2[2]
    K = hist_bhat_kernel(xi_1[0], xi_2[0])
    xty = (xi_1[1].reshape(-1, 1)*xi_2[1].reshape(1, -1)*K).sum()
    mu_part = xtx-2*xty+yty
    S = np.linalg.svd(np.dot(np.dot(xi_1[3].T, K), xi_2[3]), compute_uv=False)
    return mu_part*lambda_mu + 2*( xi_2[3].shape[1]-S.sum())






st = Scattering2D(4, (288, 352), L=4).cuda()
for dt in ['alpha', 'beta', 'gamma']:
    print('Processing split', dt)
    features = []
    labels = []
    
    for i in range(len(glob.glob('data/dyntex_' + dt + '/*'))):
        files = glob.glob('data/dyntex_' + dt+'/c'+ str(i+1) + '_*/*avi')
        Vs = []
        for f in files:
            if not ('_gr_normalized' in f):
                labels.append(i)    
                print('    Processing Video', f)
           #     V = np.load(f)
            #    features.append(np.expand_dims(V[0], 0))
                
                features.append(readvid_gr_features(f, st))
     #   print(np.concatenate(Vs).shape)
      # features.append(extract_weibull_featires(np.concatenate(Vs).reshape(-1, 3, 288, 352), st))
    
    # allfeatures = np.concatenate(features)
    # allfeatures = allfeatures.reshape(len(allfeatures), 3*features[0].shape[2], 2)[:]
    # B = -np.log(weibull_kernel(allfeatures, allfeatures))
    # nn = np.argmin(B+2*np.eye(len(allfeatures))*np.abs(B).max(), axis=1)
    # labels = np.asarray(labels)
    # print('Success rate:', np.sum(labels[nn] == np.asarray(labels))/len(allfeatures))
    # print()
    
    allfeatures = np.concatenate(features)
   # allfeatures = allfeatures.reshape(len(allfeatures), 3*features[0].shape[2], NBINS)
    B = np.zeros((len(features), len(features)))
    for i in range(B.shape[0]):
        #print(i)
        B[i] =  1- hist_bhat_kernel(allfeatures[i].reshape(1, allfeatures[i].shape[0], allfeatures[i].shape[1]), allfeatures).ravel() 
   #     for j in range(i, B.shape[0]):
    #        B[i, j] = alignment_distance(features[i], features[j], lambda_mu=0)
     #       B[j, i] = B[i, j]
    nn = np.argmin(B+2*np.eye(len(features))*np.abs(B).max(), axis=1)
    labels = np.asarray(labels)
    print('NN Success rate:', np.sum(labels[nn] == np.asarray(labels))/len(features))
    # class_centers = []
    # for i in range(labels[-1]+1):
    #     class_centers.append(class_center(list(compress(features, labels==i))))
    # B_cc = np.zeros((len(features), labels[-1]+1))
    # for i in range(B_cc.shape[0]):
    #     for j in range(B_cc.shape[1]):
    #         if labels[i]==j:
    #             cc_index = labels == j
    #             cc_index[i] = False
    #             cc = class_center(list(compress(features, cc_index)))
    #         else:
    #             cc = class_centers[j]
    #         B_cc[i, j] = alignment_distance(features[i], cc, lambda_mu=0)
    # ncc = np.argmin(B_cc, axis=1)
    # print('NCC Success rate:', np.sum(labels == ncc)/len(features))
    print()
