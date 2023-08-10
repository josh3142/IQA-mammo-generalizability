import pandas as pd 
import matplotlib.pyplot as plt

import os

import hydra 
from omegaconf import DictConfig

from typing import Optional, Tuple

def get_df(cfg: DictConfig, path) -> pd.DataFrame:
    """
    All parameters are passed with the configuration file cfg
    """
    df_name = f"trainingsresults{cfg.epoch.end}.csv"

    return pd.read_csv(os.path.join(path, df_name))

def plot_loss(df: pd.DataFrame, path: str, loss1: str, loss2: str,
    title: str, yrange: Optional[Tuple] = None) -> None:
    """
    Plot different pairs of losses for each epoch that are stored in the 
    dataframe df.
    Note that loss1, loss2 have to be columns in df. 
    """

    plt.clf()
    fig = plt.plot(df["epoch"], df[loss1], label = loss1)
    fig = plt.plot(df["epoch"], df[loss2], label = loss2)
    plt.title(title)
    if yrange is not None:
        plt.ylim(yrange)
    plt.legend(loc = "best")
    plt.savefig(os.path.join(path, f"{title}{yrange}.png"))


@hydra.main(config_path = "config", config_name = "config")
def save_fig_MSE_loss(cfg: DictConfig) -> None:
    
    df_path = f"recorded_data/{cfg.arch}"
    assert os.path.isdir(df_path), f"Folder {df_path} does not exist"

    df = get_df(cfg, df_path)
    plot_loss(df, df_path, "loss_train", "loss_test", "MSLE approximation")
    plot_loss(df, df_path, "loss_train", "loss_test", "MSLE approximation", 
        yrange = [0, 5])
    plot_loss(df, df_path, "mse_train", "mse_test", "MSE", yrange = [0, 100])
    plot_loss(df, df_path, "log_mse_train", "log_mse_test", "MSLE", 
        yrange = [0, 100])
   
if __name__ == "__main__":
    save_fig_MSE_loss()
    
