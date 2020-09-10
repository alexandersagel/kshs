#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:12:25 2020

@author: Alexander Sagel
"""

from kymatio.torch import Scattering2D
import imageio
import numpy as np
import glob
import os
import torch
import tqdm

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

global NBINS
NBINS = 20


def readvid_gr_features(filename, st, normalized=None):
    '''
    Loads video from a file, converts it to Grayscale and computes a sequences
    of Scattering subband histogram vectors

    Parameters
    ----------
    filename : String,
        Location of Dynamic Texture Video.
    st : Kymatio Torch Module,
        2D Scattering transform model.
    normalized : Tuple, optional
        Scattering parameters (J, L) if the Scattering coefficients are to be
        normalized. The default is None.

    Returns
    -------
    feature_sequence : Numpy Array,
        Sequence of subband histogram features (N x S x B)
        N=Video sequence length, S=# of Scattering channels
        B=NBINS=# of Histogram bins.
    '''
    V = []
    vid = imageio.get_reader(filename,  'ffmpeg')
    for image in vid.iter_data():
        arr = np.float32(np.asarray(image))/255.0
        grarr = 0.299*arr[:, :, 0] + 0.587*arr[:, :, 1] + 0.114*arr[:, :, 2]
        V.append(extract_hist_features(np.expand_dims(grarr, 0), st,
                                       normalized=normalized))
    vid.close()
    feature_sequence = np.concatenate(V).reshape(len(V), -1, NBINS)
    return feature_sequence


def extract_hist_features(imgs, st, normalized):
    '''
    Expects a feature vector of Scattering subband histograms from an image
    tensor

    Parameters
    ----------
    imgs : Numpy Array,
        Image array (N x C x H x W) or (C x H x W).
    st : Kymatio Torch Module,
        2D Scattering transform model.
    normalized : Tuple, optional
        Scattering parameters (J, L) if the Scattering coefficients are to be
        normalized. Otherwise None.

    Returns
    -------
    features : Numpy Array
        Subband histogram features (N x C x S x B), S=# of Scattering channels
        B=NBINS=# of Histogram bins.
    '''
    imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
    stimgs = st(torch.tensor(np.float32(imgs), device='cuda')).cpu().numpy()
    nchannels = stimgs.shape[-3]
    features = np.zeros((imgs.shape[0], imgs.shape[1], nchannels, NBINS))
    for i in range(len(imgs)):
        for j in range(imgs.shape[1]):
            for l in range(nchannels):
                if normalized is not None:
                    J, L = normalized
                    bns = np.arange(NBINS+1)/NBINS
                    if l == 0:
                        smpls = stimgs[i, j, l]
                    elif l < L*J+1:
                        smpls = stimgs[i, j, l]/np.abs(imgs[i, j]).mean()
                    else:
                        smpls = stimgs[i, j, l]/(stimgs[i, j, 2*(l-L*J-1)
                                                        // (L*(J-1))+1]+1e-16)
                else:
                    if l == 0:
                        bns = np.arange(NBINS+1)/NBINS
                    else:
                        bns = np.arange(NBINS+1)/(NBINS*16)
                    smpls = stimgs[i, j, l]
                h, _ = np.histogram(smpls, bns, range=(bns[0], bns[-1]))
                h[-1] += (smpls > bns[-1]).sum()
                features[i, j, l] = h/h.sum()
    return features


st = Scattering2D(4, (288, 352), L=4).cuda()

for dt in ['beta', 'gamma', 'alpha']:
    print('Processing split', dt)
    features = []
    labels = []
    for i in tqdm.tqdm(range(len(glob.glob('data/dyntex_' + dt + '/*')))):
        files = glob.glob('data/dyntex_' + dt + '/c' + str(i+1) + '_*/*.avi')
        for f in files:
            labels.append(i)
            #Normalized Scattering Transform:
            np.save(f[:-4] + '_nst.npy', readvid_gr_features(f, st, (4, 4)))
            #Regular Scattering Transform
            np.save(f[:-4] + '_st.npy', readvid_gr_features(f, st))
