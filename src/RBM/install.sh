#!/bin/sh
sudo apt-get update
sudo apt install python-minimal -y
sudo apt-get install python-setuptools python-dev build-essential -y
sudo easy_install pip
sudo -H pip install -- upgrade pip 
sudo -H pip install numpy 
sudo -H pip install pandas
sudo -H pip install scipy
sudo -H pip install sklearn
sudo apt-get update
