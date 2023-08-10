# needed for deterministic behaviour
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time

import hydra 
from omegaconf import DictConfig 
from pathlib import Path

from model import choose_linear_model
from net import Net, get_msle_approximation
from dataset import get_dataset, data_transformation
from utils import create_df, save_df, get_encoder

import numpy as np
import random


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    # set seed and make operations deterministic
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only = True)
    torch.set_num_threads(16)

    # create folder to store data 
    print(f"All parameters: {cfg}")
    print("")

    # create folder to store data
    path = f"recorded_data/{cfg.arch}"
    Path(path).mkdir(parents = True, exist_ok = True)
    Path(path + "/checkpoints").mkdir(parents = True, exist_ok = True)

    # build model
    encoder, n_input = get_encoder(cfg.arch, cfg.device)        
    linear_model = choose_linear_model(
        n_input, 
        predict_coeff = cfg.optim.predict_coeff)
    if cfg.optim.pretrain_full:
        checkpoint   = torch.load(path + cfg.checkpoints.model.path_load, 
            map_location = cfg.device)
        print(f'checkpoint is stored at epoch: {checkpoint["epoch"]}')
        if checkpoint["epoch"] == cfg.epoch.start:
            model = nn.Sequential(encoder, choose_linear_model(n_input, False))
            print("Pretrained model is loaded")
            model.load_state_dict(checkpoint["state_dict"])
            encoder = model[0]
            if cfg.optim.train_all_layer:
                model = nn.Sequential(encoder, linear_model)
            else:
                model = linear_model
        else:
            raise ValueError("Loaded net starts from wrong epoch")
    else:
        if cfg.optim.train_all_layer:
            model = nn.Sequential(encoder, linear_model)
        else:
            model = linear_model

   # initialize objective, optimizer and net class,
    init_lr   = cfg.optim.lr * cfg.optim.batch_size / 256   
    objective = lambda y_hat, y: get_msle_approximation(y, y_hat)
    optimizer = torch.optim.Adam(model.parameters(), init_lr,
            betas        = (cfg.optim.adam.beta1, cfg.optim.adam.beta2),
            eps          = cfg.optim.adam.eps,
            weight_decay = cfg.optim.weight_decay)                
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, 
        gamma = 0.5, last_epoch = - 1, verbose = False)
    net = Net(model, backbone = encoder, optimizer = optimizer, 
        objective = objective, device = cfg.device)

    # initialize dataset and dataloader
    data_train = get_dataset(cfg.data.name, cfg.data.path_train,
            types = ["IMA", "dcm"],
            margins = cfg.margins,
            transform = data_transformation(cfg.data.name, cfg.arch, 
                cfg.data.size, are_imgs_augmented = True) 
            )
    dl_train   = DataLoader(data_train, batch_size = cfg.optim.batch_size,
        shuffle = True, num_workers = cfg.optim.n_workers)
    
    data_test  = get_dataset(cfg.data.name, cfg.data.path_test, 
            types = ["IMA", "dcm"],
            margins = None,
            transform = data_transformation(cfg.data.name, cfg.arch, 
                    cfg.data.size, are_imgs_augmented = False) )
    dl_test    = DataLoader(data_test, batch_size = cfg.optim.batch_size,
        shuffle = False, num_workers = cfg.optim.n_workers)
        
    print(f"Dataset size: {len(data_train)}.")
    img_shape_train = data_train.__getitem__(0)[0].shape
    img_shape_test  = data_test.__getitem__(0)[0].shape 
    print(f"Image size (train): {img_shape_train}")
    print(f"Image size (test): {img_shape_test}")
    assert img_shape_test == img_shape_train, "Train and test image have " + \
        "different shape"

    # train model, store checkpoints
    losses_train, mses_train = [], []
    losses_test, mses_test = [], []
    log_mses_train, log_mses_test = [], []
    lrs = []
    print("epoch \t loss_train \t mse_train \t loss_test \t mse_test \t" +
        "t_train \t t_test")
    for epoch in range(cfg.epoch.start, cfg.epoch.end):
        start_time = time.time()
        loss_train, mse_train = net.train_one_epoch(dl_train, 
            cfg.optim.train_all_layer)
        scheduler.step()
        train_time = time.time() - start_time
        start_time = time.time()
        loss_test, mse_test   = net.test_one_epoch(dl_test, 
            cfg.optim.train_all_layer)
        test_time = time.time() - start_time
            
        # Store and print all results
        loss_train = np.round(loss_train, 4)
        mse_train  = np.round(mse_train * 100, 2)
        loss_test  = np.round(loss_test, 4)
        mse_test  =  np.round(mse_test * 100, 2)

        print(f"{epoch}\t {loss_train:.2f}\t\t {mse_train:.2f}%\t"
            + f" {loss_test:.2f}\t\t {mse_test:.2f}%\t"
            + f" {train_time:.2f}\t\t {test_time:.2f}")
            
        log_mse_train = net.test_one_epoch_log(dl_train, 
            cfg.optim.train_all_layer)
        log_mse_test  = net.test_one_epoch_log(dl_test, 
            cfg.optim.train_all_layer)
        log_mse_train = np.round(log_mse_train, 2)
        log_mse_test = np.round(log_mse_test, 2)
        
        log_mses_train.append(log_mse_train)
        log_mses_test.append(log_mse_test) 
        losses_train.append(loss_train)
        mses_train.append(mse_train)
        losses_test.append(loss_test)
        mses_test.append(mse_test)

        current_lr = scheduler.get_last_lr()[0]
        if not len(scheduler.get_last_lr()) == 1:
            print("The correct learning rate might not be displayed.")
        lrs.append(current_lr)
        
        # store model
        if cfg.optim.train_all_layer:
            model_save = net.model 
        else:
            model_save = nn.Sequential(net.backbone, net.model)
            
        if cfg.checkpoints.model.save and (epoch == (cfg.epoch.end - 1) or 
            epoch > cfg.epoch.end // 2 and epoch%cfg.epoch.frequ_to_save == 0):
             net.save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg.arch,
                'state_dict': model_save.state_dict(),
                'optimizer' : net.optimizer.state_dict(),
                'loss_test': loss_test,
                'mse_test': mse_test
            }, filename = path + f"/checkpoints/model{epoch:04d}.pth.tar")            
                    
        # store data in each epoch (is_file_overwritten should be True in this case)
        df = create_df(losses_train, mses_train, losses_test, mses_test, 
            log_mses_train, log_mses_test, lrs,
            start_epoch = cfg.epoch.start)

        df_name = f"trainingsresults{cfg.epoch.end}.csv"
        save_df(df, path + "/" + df_name, is_file_overwritten = True)

if __name__ == "__main__":
    run_main()