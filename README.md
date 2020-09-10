# Dynamic Texture Recognition via Nuclear Distances on Kernelized Scattering Histogram Spaces
Code for reproducing the experiments from the paper "Dynamic Texture Recognition via Nuclear Distances on Kernelized Scattering Histogram Spaces"


### Running the Experiments

1. (a) Either run `python download_scattering_histograms.py` to download the Scattering histogram sequences

*or*

1. (b) Download the Dynamic Texture videos as avi files from http://dyntex.univ-lr.fr/ and place them into the according directories in the `data` folder and run `python extract_histograms_from_videos.py`. This can take some time and requires Kymatio (https://www.kymat.io/) on CUDA

2. Run `python evaluate_kshs.py` and `python evaluate_knshs.py` to perform the evaluation
