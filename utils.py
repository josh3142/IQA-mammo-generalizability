from torch import nn

import pandas as pd
from typing import List, Tuple

from model import get_backbone


def create_df(loss_train: List[float], mse_train: List[float], 
    loss_test: List[float], mse_test: List[float], log_mses_train: List[float], 
    log_mses_test: List[float], lrs: List[float], 
    start_epoch: int = 0) -> pd.DataFrame:
    df = pd.DataFrame(data =
            {"epoch": [int(i) + start_epoch for i in range(len(loss_train))],
            "loss_train" : loss_train,
            "mse_train": mse_train,
            "loss_test": loss_test,
            "mse_test": mse_test,
            "log_mse_train": log_mses_train,
            "log_mse_test": log_mses_test,
            "lr": lrs
            }
        )
    return df
    
def save_df(df: pd.DataFrame, filename: str, 
    is_file_overwritten: bool = False) -> None:
    
    if not is_file_overwritten:
        try:
            df_loaded = pd.read_csv(filename)
            df = pd.concat([df_loaded, df], axis = 0)
        except FileNotFoundError:
            print("File does not exists.")
            print("A new file is created")
    df.to_csv(filename, index = False)


def get_encoder(arch: str, device: str) -> Tuple[nn.Module, int]:
    """
    Gives the encoder (everything but the last layer of the model) and the 
    input dimension n_input for the last layer
    """
    encoder = get_backbone(arch).to(device)
    if "resnet" in arch:
        n_input    = encoder.fc.in_features
        encoder.fc = nn.Identity()
    elif "densenet" in arch:
        n_input            = encoder.classifier.in_features
        encoder.classifier = nn.Identity()
    elif "efficient" in arch:
        n_input            = encoder.classifier[1].in_features
        encoder.classifier = nn.Identity()
    elif ("vgg" in arch) or ("cnn" in arch):
        n_input               = encoder.classifier[6].in_features
        encoder.classifier[6] = nn.Identity()        
    return encoder, n_input