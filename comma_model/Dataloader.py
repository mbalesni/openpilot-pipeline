import numpy as np 
from numpy import load
import torch 
import os 
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms

"""
To do : Add intial calib transformations, assertions for tensor shapes 
"""
class CommaLoader(Dataset):

    def __init__(self, npz_paths, dummy_test = None, transform = None, device = None):
        """
        Dataloader for Comma model train. pipeline

        Summary: 
            This dataloader can be used for intial testing and for proper training
            Images are converted into YUV 4:2:0 channels and brought to a calib frame of reff 
            as used in the official comma pipeline.
        
        Args: ------------------

        """

        self.dummy_test = dummy_test
        self.npz_paths = npz_paths 
        self.transform = transform
        self.device = device 

        if dummy_test:
            self.input = np.load(self.npz_paths[0])
            self.gt = np.load(self.npz_paths[1])  
        

        
    def __len__(self):

        if self.dummy_test:
            return len(self.input)

        else: 
            pass

    def __getitem__(self, index):
        
        if self.dummy_test:
            imgs = self.input['arr_0'][index]


            
        else:
            # add transformations 
            pass 
        
        return 
    

class Transformations():
    
    def __init__(self):
    
        pass 

    def InitialImageTransform(self):
        
        pass 

    def ArrayToTensor(self):
        
        pass 


numpy_paths = ["inputdata.npz","gtdata.npz"]