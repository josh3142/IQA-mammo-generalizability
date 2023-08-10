#!/usr/bin/bash

device="cuda:0"
data="lanczos"
lr=0.0004
epoch_end=10
load_model="/checkpoints/model0009.pth.tar"
arch="cnnb3bn,efficientnets,resnet18,densenet121,vgg13bn"
ext="eps"
seed=1

python run.py -m device=${device} data=${data} optim.lr=${lr} \
  epoch.end=${epoch_end} arch=${arch} seed=${seed}
python plot_loss.py -m device=${device} data=${data} optim.lr=${lr} \
  epoch.end=${epoch_end} arch=${arch} seed=${seed}
python create_CDCs.py -m device=${device} data=${data} optim.lr=${lr} \
  epoch.end=${epoch_end} arch=${arch} checkpoints.model.path_load=${load_model} \
  seed=${seed}
