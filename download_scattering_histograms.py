#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:54:54 2020

@author: Alexander Sagel
"""

import urllib.request
import zipfile
import os
import shutil

print('Downloading scattering_histograms.zip...')
url = 'https://www.dropbox.com/s/7m3tmohqyi9fod1/scattering_histograms.zip?dl=1'
urllib.request.urlretrieve(url, './data/scattering_histograms.zip')
print('Unzipping...')
zf = zipfile.ZipFile('./data/scattering_histograms.zip')
for f in zf.namelist():
    if f.endswith('npy'):
        zf.extract(f, 'data/')
        p = f.split('/')
        os.rename('data/'+f, 'data/'+p[1]+'/'+p[2] + '/' + '/' + p[3])
shutil.rmtree('data/'+p[0])
print('Downloading scattering_histograms_normalized.zip...')
url = 'https://www.dropbox.com/s/dquj7cd368wmzzl/scattering_histograms_normalized.zip?dl=1'
urllib.request.urlretrieve(url, './data/scattering_histograms_normalized.zip')
print('Unzipping...')
zf = zipfile.ZipFile('./data/scattering_histograms_normalized.zip')
for f in zf.namelist():
    if f.endswith('npy'):
        zf.extract(f, 'data/')
        p = f.split('/')
        os.rename('data/'+f, 'data/'+p[1]+'/'+p[2] + '/' + '/' + p[3])
shutil.rmtree('data/'+p[0])
print('Done!')