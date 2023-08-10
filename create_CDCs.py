import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from typing import Optional, Tuple, List

import hydra
from omegaconf import DictConfig 

import os
from pathlib import Path

from dataset import get_dataset, data_transformation
from model import last_layer
from utils import get_encoder


def load_model(path: str, arch: str, n_class: int, device: str) -> nn.Module:
    """
    Loads the model of type arch with n_class classes from path. The 
    loaded dictionary has to have the parameters of the model stored with key
    "state_dict"
    """
    checkpoint   = torch.load(path, map_location = device)

    # build model
    encoder, n_input = get_encoder(arch, device) 
    linear_model = last_layer(n_input = n_input,
                            n_classes = n_class)
    model        = nn.Sequential(encoder, linear_model).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    return model

def get_df_results(imgs: List[int],  
    diameters: List[int], Ys: List[float], Ys_hat: List[float],
    log10Ys: List[float], log10Ys_hat: List[float]) -> None:
    """
    Store all the data that is relevant for data analysis
    """
    df = pd.DataFrame(data = {
            "img": imgs,
            "diameter": diameters,
            "y": Ys,
            "y_hat": Ys_hat,
            "log10_y": log10Ys,
            "log10_y_hat": log10Ys_hat
        }, index = None)
    df.sort_values(by = ["img", "diameter"], inplace = True)

    return df


def plot_cdc(Y: Tensor, Y_hat: Tensor, std: Optional[Tensor], 
        path: str, extension: str, 
    	xlim: Optional[Tuple[float, float]] = None, 
        ylim: Optional[Tuple[float, float]] = None,
        save_fig: bool = True) -> None:
    """
    Plot contrast-detail curve prediction and ground truth
    """
    diameter = np.array([0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.31, 
        0.40, 0.50, 0.63, 0.80, 1.00])
    plt.clf()
    plt.plot(diameter, Y, "black", label = 'ground truth', marker = 'X')
    plt.plot(diameter, Y_hat, 'red', label = 'prediction', marker = "o")
    if std is not None:
        plt.fill_between(diameter, Y_hat - std, Y_hat + std, color = "red", 
            alpha = 0.25, label = "std deviation")

    plt.grid('on')
    plt.yscale("log")
    plt.xlabel('diameter [mm]', fontsize = 15)
    plt.ylabel('threshold [\N{greek small letter mu}m]', fontsize = 15)
    plt.title("CDC", fontsize=25)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    
    if save_fig:
        plt.savefig(os.path.join(path, f"CDC.{extension}"))
    
    
@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    path = f"recorded_data/{cfg.arch}"
    plot_path = os.path.join(path, "CDC") 
    Path(plot_path).mkdir(parents = True, exist_ok = True)

    # load model
    checkpoint_path = path + cfg.checkpoints.model.path_load
    model = load_model(checkpoint_path, cfg.arch, cfg.data.n_class, cfg.device)
    
    # load dataset
    data = get_dataset(cfg.data.name, cfg.data.path_test,
            types = ["IMA", "dcm"],
            margins = cfg.margins,
            transform = data_transformation(cfg.data.name, cfg.arch, 
                cfg.data.size, are_imgs_augmented = True))
    dl  = DataLoader(data, batch_size = cfg.optim.batch_size,
        shuffle = False, num_workers = 0)

    # create columns for df, initialize df
    imgs, diameters = [], []
    Ys, Ys_hat, log10Ys, log10Ys_hat = [], [], [], []
    diameters_values = np.array([0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.31, 
        0.40, 0.50, 0.63, 0.80, 1.00])

    # generate dataframe 
    model.eval()
    with torch.no_grad():
        for idx_batch, (X, Y) in enumerate(dl):
            X, Y = X.to(cfg.device, dtype = torch.float), Y.to(cfg.device)
            Y_hat      = model(X)
            Y_hat = Y_hat.type(torch.cuda.FloatTensor) 
            Y     = Y.type(torch.cuda.FloatTensor)
            log10Y     = torch.log10(Y)
            log10Y_hat = torch.log10(Y_hat)
            for n in range(len(Y)):
                y, y_hat           = Y[n], Y_hat[n]
                log10y, log10y_hat = log10Y[n], log10Y_hat[n]
                for d in range(len(y)):
                    imgs.append(n + idx_batch * cfg.optim.batch_size)
                    diameters.append(diameters_values[d]) 
                    Ys.append(y[d].item()) 
                    Ys_hat.append(y_hat[d].item())
                    log10Ys.append(log10y[d].item()) 
                    log10Ys_hat.append(log10y_hat[d].item())
    assert len(torch.unique(Y)) == Y.shape[-1], ("Not all images are " +
        "from the same scanner")

    df = get_df_results(imgs, diameters, Ys, Ys_hat, log10Ys, log10Ys_hat)

    Y     = df[df["img"] == 0].y
    Y_hat = df.groupby(["diameter"], as_index = False).mean()["y_hat"]
       
    # generate plot (Y, Y_hat) and store data
    plot_cdc(Y = Y, Y_hat = Y_hat, std = None,
        path = plot_path, extension = cfg.plot.extension,
        ylim = (1e-3, 10)) 
    
if __name__ == "__main__":
    run_main()

        
        