#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:54:54 2020

@author: Alexander Sagel
"""

import urllib.request
import zipfile

print('Downloading scattering_histograms.zip...')
url = 'https://www.dropbox.com/s/7m3tmohqyi9fod1/scattering_histograms.zip?dl=1'
urllib.request.urlretrieve(url, './data/scattering_histograms.zip')
print('Unzipping...')
zf = zipfile.ZipFile('./data/scattering_histograms.zip')
zf.extractall('./data/')
print('Downloading scattering_histograms_normalized.zip...')
url = 'https://www.dropbox.com/s/dquj7cd368wmzzl/scattering_histograms_normalized.zip?dl=1'
urllib.request.urlretrieve(url, './data/scattering_histograms_normalized.zip')
print('Unzipping...')
zf = zipfile.ZipFile('./data/scattering_histograms_normalized.zip')
zf.extractall('./data/')
print('Done!')