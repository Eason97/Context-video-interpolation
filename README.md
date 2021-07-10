# Context-video-interpolation
This is the implementation of  video interpolation, which is based on optical flow to estimate motion information and backwarp warping to interpolate middle frame. In addition, context feature is used to into the interpolation network. We use [Vimeo 90K](http://toflow.csail.mit.edu/)  for training and testing.

## Environment:

- Ubuntu: 18.04

- CUDA Version: 11.0 
- Python 3.8

## Dependencies:

- torch==1.6.0
- torchvision==0.7.0
- NVIDIA GPU and CUDA

