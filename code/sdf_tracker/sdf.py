from abc import ABC, abstractclassmethod
from dataclasses import dataclass
import torch
import numpy as np

# might not need it, can be add the VolumetricGridSdf
# in case more type of tracking method will be implemented

class Sdf(ABC):
    def __init__(
        self,
        T,
        device,
        counter = 0,
        z_max = 3.5,
        z_min = 0.5
        ):

        self.T = T
        self.inv_T = 1 / T
        
        self.device = device

        self.counter = counter

        self.z_max = z_max
        self.z_min = z_min
        
    def truncate(self, sdf):
        sdf = torch.where(sdf>-self.T, sdf, torch.Tensor([-self.T]).to(self.device))
        sdf = torch.where(sdf<self.T, sdf, torch.Tensor([self.T]).to(self.device))
        return sdf

    def weight(self, sdf):
        w = torch.zeros_like(sdf)
        epsilon = 0.5*self.T

        w = torch.where(sdf>=-self.T, 1.0+sdf*self.inv_T, w)
        w = torch.where(sdf>=0, torch.Tensor([1.]).to(self.device), w)

        # print("weight shape:", w.shape)
        
        return w

    def update_counter(self):
        self.counter += 1
        
    @abstractclassmethod
    def weights(self, points):
        pass

    @abstractclassmethod
    def update(self, rgb, depth, K, pose, Nest):
        pass

    @abstractclassmethod
    def setup(self, rgb, depth, K, Nest):
        pass


    @abstractclassmethod
    def export_pc(self, filename):
        pass