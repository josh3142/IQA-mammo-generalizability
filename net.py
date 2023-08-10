import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn, optim

from typing import Tuple, Dict
from torch import Tensor
import numpy as np


def get_msle_approximation(y: Tensor, y_hat: Tensor, beta: float = 3) -> Tensor:
    """ 
    Approximates the msle of the prediction y_hat and true value y:
        log(y_hat / y) = log(y_hat) - log(y).
    Since y_hat and y can have different sign, the argument of the log is 
    forced to be positive with softplus:
        log(softplus(y_hat / y))
    Note: 
        Lower values of beta lead to easier training. 
        reg is added to avoid ln(0).
    """
    reg         = 1.e-20
    min_shift  = 1 / beta * np.log(np.exp(beta) - 1) - 1 
    softplus   = nn.Softplus(beta = beta)(y_hat / y + min_shift)
    msle_reg = torch.log(softplus + reg)**2
    return msle_reg.sum()


class Net():
    """
    Supplements a pytorch model with useful functions e.g. training and test.
    The class can either train an entire model from scratch, in which case the
    argument backbone is unimportant, or it can be used to train the last layer 
    only
    """
    def __init__(self, model: nn.Module, backbone: nn.Module, 
        optimizer: optim.Optimizer, objective: nn.Module, 
        device: str = "cpu") -> None:
        self.model     = model.to(device)
        self.backbone  = backbone.to(device)
        self.optimizer = optimizer
        self.objective = objective
        self.device    = device
        

    def __call__(self, *args) -> nn.Module:
        return self.model(*args)
        

    def get_loss(self, X: Tensor, Y: Tensor, train: bool = True) -> Tuple[
        float, float]:

        mu    = self.model(X)
        mu, Y = mu.type(torch.cuda.FloatTensor), Y.type(torch.cuda.FloatTensor)
        loss  = self.objective(mu, Y)
        if train == True:
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, mu 
    

    def get_se(self, Y: Tensor, Y_hat: Tensor) -> float:
        "Calculates the sqaured error"
        return ((Y - Y_hat)**2).sum().item()
        

    def get_msle(self, Y: Tensor, Y_hat: Tensor) -> float:
        """
        This is the ideal loss function. Hence if this value is well definied 
        it should be the scalar that characterizes the quality of the prediction
        best.
        """        
        return (torch.log(Y_hat / Y)**2).sum().item()    


    def train_one_epoch(self, dl: DataLoader, train_all_layers: bool) -> Tuple[
            float, float]:

        self.model.train()
        loss_sum, se, n_elements = 0, 0, 0
        for X, Y in dl:
            X, Y = X.to(self.device, dtype = torch.float), Y.to(self.device)
            if not train_all_layers:
                X = self.backbone(X)
            loss, Y_hat = self.get_loss(X, Y)
            loss_sum   += loss.item()
            se         += self.get_se(Y, Y_hat)
            n_elements += Y.shape[0]
        mse       = se / n_elements
        loss_mean = loss_sum / n_elements 
        
        return loss_mean, mse    
        

    def test_one_epoch(self, dl: DataLoader, train_all_layers: bool) -> Tuple[
            float, float]:

        self.model.eval()
        loss_sum, se, n_elements = 0, 0, 0
        with torch.no_grad():
            for X, Y in dl:
                X, Y = X.to(self.device, dtype = torch.float), Y.to(self.device)
                if not train_all_layers:
                    X = self.backbone(X)
                loss, Y_hat = self.get_loss(X, Y, train = False)
                loss_sum   += loss.item()
                se         += self.get_se(Y, Y_hat)
                n_elements += Y.shape[0]
            mse       = se / n_elements
            loss_mean = loss_sum / n_elements
                
        return loss_mean, mse


    def test_one_epoch_log(self, dl: DataLoader, 
        train_all_layers: bool) -> Tuple[float, float]:

        self.model.eval()
        msle, n_elements = 0, 0
        with torch.no_grad():
            for X, Y in dl:
                X, Y = X.to(self.device, dtype = torch.float), Y.to(self.device)
                if not train_all_layers:
                    X = self.backbone(X)
                Y_hat       = self.model(X)
                msle     += self.get_msle(Y, Y_hat)
                n_elements += Y.shape[0]
            log_mse   = msle / n_elements
                
        return log_mse


    def save_checkpoint(self, state: Dict, 
        filename: str = 'checkpoint.pth.tar') -> None:
        
        torch.save(state, filename)