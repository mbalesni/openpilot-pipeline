import numpy as np 
from numpy import load
import torch 
import os 
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms

"""
To do : Add intial calib transformations, assertions for tensor shapes,
        efficient strategy for inputs and outputs __getitem__ 
"""
class CommaLoader(Dataset):

    def __init__(self, npz_paths, dummy_test = None, transform = None, device = None):
        super(CommaLoader, self).__init__()
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

        if self.npz_paths is None:
            raise TypeError("add dummy data paths")
        
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
            
            imgs = torch.from_numpy(self.input['arr_0'][index]).to(self.device)
            desire = torch.from_numpy(self.input['arr_1'][index]).to(self.device)
            traffic_conv = torch.from_numpy(self.input['arr_2'][index]).to(self.device)
            recurrent_state = torch.from_numpy(self.input['arr_3'][index]).to(self.device)
            
            plan = torch.from_numpy(self.gt['arr_0'][index]).to(self.device)
            ll = torch.from_numpy(self.gt['arr_1'][index]).to(self.device)
            ll_prob = torch.from_numpy(self.gt['arr_2'][index]).to(self.device)
            road_edges = torch.from_numpy(self.gt['arr_3'][index]).to(self.device)
            leads = torch.from_numpy(self.gt['arr_4'][index]).to(self.device)
            leads_prob = torch.from_numpy(self.gt['arr_5'][index]).to(self.device)
            desire_gt = torch.from_numpy(self.gt['arr_6'][index]).to(self.device)
            meta_various = torch.from_numpy(self.gt['arr_7'][index]).to(self.device)
            meta_desire = torch.from_numpy(self.gt['arr_8'][index]).to(self.device)
            pose = torch.from_numpy(self.gt['arr_9'][index]).to(self.device)

            return (imgs, desire, traffic_conv, recurrent_state), (plan,
            ll, ll_prob, road_edges, leads, leads_prob, desire_gt, meta_various,
            meta_desire, pose)
        
        else:
            # add transformations 
            pass 
        
        

class Transformations():
    
    def __init__(self):
    
        pass 

    def InitialImageTransform(self):
        
        pass 

    def ArrayToTensor(self):
        
        pass 


if __name__ == "__main__":

    numpy_paths = ["inputdata.npz","gtdata.npz"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    comma_data = CommaLoader(numpy_paths, dummy_test= True)
    comma_loader = DataLoader(comma_data, batch_size=1, shuffle=True)

    for i, data in enumerate(comma_loader):
        print(data.shape)