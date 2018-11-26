#!/bin/sh
sudo apt-get update
sudo apt install python-minimal -y
sudo apt-get install python-setuptools python-dev build-essential -y
sudo easy_install pip
sudo -H pip install -- upgrade pip 
sudo -H pip install numpy 
sudo -H pip install pandas
sudo apt-get update
sudo apt-get install python-pip python-dev -y
# sudo -H pip install tensorflow-gpu
# wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb"
# sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
# sudo apt-get update
# sudo apt-get install cuda-9.0 -y
# sudo tar -xvf cudnn-9.1-linux-x64-v7.1.tgz -C /usr/local
# export PATH=/usr/local/cuda/bin:$PATH
# export  LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
# export CUDA_HOME=/usr/local/cuda
# sudo ldconfig /usr/local/cuda/lib64