from torch.utils.data import Dataset 
import torch
from torch import Tensor

import numpy as np

import h5py

from typing import Tuple, Optional, Callable, List


class Mammo_h5(Dataset):
    """
    Reads a dataset of the h5-File.
    Margins can be selected. If margins == None, data of all margins from 
    the h5-file is taken.
    """
    def __init__(self, in_file: str, transform: Callable = None, 
        margins: Optional[List[int]] = None) -> None:

        super().__init__()
        self.file_path   = in_file
        self.margins_sel = margins
        self.imgs, self.labels = self.get_data()
        self.len       = len(self.imgs)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.imgs[idx]
        y = self.labels[:, idx]

        if self.transform:
            x = self.transform(x)

        return x, y


    def select_data_simulated(self, imgs_sim: Tensor, labels_sim: Tensor, 
        margins_sim: Tensor) -> Tuple[Tensor]:
        "Selects data with the correct margin."

        imgs_filtered   = []
        labels_filtered = []
        for margin in self.margins_sel:
            mask = (margins_sim == margin).flatten()
            imgs_filtered.append(imgs_sim[mask, :, :])
            labels_filtered.append(labels_sim[mask, :])
        imgs_filtered   = torch.vstack(imgs_filtered)
        labels_filtered = torch.vstack(labels_filtered)
        
        return imgs_filtered, labels_filtered
        
    def get_data(self) -> Tuple[Tensor]:
        """
        Extract the images and labels in Tensors.
        For simulated data a margin might be added
        """
        dataset     = h5py.File(self.file_path, 'r')
        
        # gets real data
        if "real" in dataset.keys():
            imgs_real, labels_real = self.get_data_real(dataset)
            labels_real = labels_real.T
        else:
            imgs_real, labels_real = torch.Tensor([]), torch.Tensor([])
            
        # gets simulated data
        if "simulation" in dataset.keys():
            imgs_sim, labels_sim, margins_sim = self.get_data_simulated(dataset)
            if self.margins_sel is not None:
                imgs_sim, labels_sim = self.select_data_simulated(imgs_sim, 
                    labels_sim, margins_sim)
            labels_sim = labels_sim.T
        else:
            imgs_sim, labels_sim = torch.Tensor([]), torch.Tensor([])
            
        imgs   = torch.cat((imgs_real, imgs_sim), 0)
        labels = torch.cat((labels_real, labels_sim), 1)
        
        assert imgs.size(0) == (imgs_real.size(0) + imgs_sim.size(0)), \
            "Not all selected images are included in the dataset"

        return imgs, labels


    def get_data_simulated(self, dataset: Dataset) -> Tuple[Tensor]:
        imgs_sim    = torch.Tensor(np.array(dataset['simulation/images']).
            astype('float32'))
        labels_sim  = torch.Tensor(np.array(dataset['simulation/labels'])).T
        margins_sim = torch.Tensor(np.array(dataset['simulation/margins'])).T

        return imgs_sim, labels_sim, margins_sim
        

    def get_data_real(self, dataset: Dataset) -> Tuple[Tensor]:
        imgs_real   = torch.Tensor(np.array(dataset['real/images'])
            .astype('float32'))
        labels_real = torch.Tensor(np.array(dataset['real/labels'])).T

        return imgs_real, labels_real
        

    def __len__(self) -> int:
        return self.len