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

    def __init__(self, npz_paths, split_per, dummy_test = False, transform = None, device = None, train= False, test = False):
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
        self.train = train
        self.test = test
        self.split_per = split_per

        if self.npz_paths is None:
            raise TypeError("add dummy data paths")
        
        if dummy_test:
            self.input = np.load(self.npz_paths[0])
            self.gt = np.load(self.npz_paths[1])
        
        else:
            pass ## conditions for real data
    
    def split_data(self, input_data):
        
        numberofsamples = input_data['imgs'][0].shape
        length_data = numberofsamples[0]
        self.split_index = int(np.round(length_data) * self.split_per)
        
        if self.train:
            len_for_loader = self.split_index 
            return len_for_loader   
        
        elif self.test:
            len_for_loader = length_data - self.split_index
            return len_for_loader

    def __len__(self):

        if self.dummy_test:
            sample_length = self.split_data(self.input)
            return sample_length
        
        else: # for real data 
            pass

    def __getitem__(self, index):
        
        if self.dummy_test and self.train:
            
            imgs = torch.from_numpy(self.input['imgs'][0][index]).to(self.device)
            desire = torch.from_numpy(self.input['desire'][0][index]).to(self.device)
            traffic_conv = torch.from_numpy(self.input['traff_conv'][0][index]).to(self.device)
            
            #need only during intialisation
            recurrent_state = torch.from_numpy(self.input['recurrent_state'][0][index]).to(self.device)
            
            plan = torch.from_numpy(self.gt['plan'][0][index]).to(self.device)
            ll = torch.from_numpy(self.gt['ll'][0][index]).to(self.device)
            ll_prob = torch.from_numpy(self.gt['ll_prob'][0][index]).to(self.device)
            road_edges = torch.from_numpy(self.gt['road_edges'][0][index]).to(self.device)
            leads = torch.from_numpy(self.gt['leads'][0][index]).to(self.device)
            leads_prob = torch.from_numpy(self.gt['leads_prob'][0][index]).to(self.device)
            desire_gt = torch.from_numpy(self.gt['desire'][0][index]).to(self.device)
            meta_various = torch.from_numpy(self.gt['meta_various'][0][index]).to(self.device)
            meta_desire = torch.from_numpy(self.gt['meta_desire'][0][index]).to(self.device)
            pose = torch.from_numpy(self.gt['pose'][0][index]).to(self.device)

            return (imgs, desire, traffic_conv, recurrent_state), (plan,
            ll, ll_prob, road_edges, leads, leads_prob, desire_gt, meta_various,
            meta_desire, pose)
            
        elif self.dummy_test and self.test:
            imgs = torch.from_numpy(self.input['imgs'][0][self.split_index+index]).to(self.device)
            desire = torch.from_numpy(self.input['desire'][0][self.split_index+index]).to(self.device)
            traffic_conv = torch.from_numpy(self.input['traff_conv'][0][self.split_index+index]).to(self.device)
            
            #need only during intialisation
            recurrent_state = torch.from_numpy(self.input['recurrent_state'][0][self.split_index+index]).to(self.device)
            
            plan = torch.from_numpy(self.gt['plan'][0][self.split_index+index]).to(self.device)
            ll = torch.from_numpy(self.gt['ll'][0][self.split_index+index]).to(self.device)
            ll_prob = torch.from_numpy(self.gt['ll_prob'][0][self.split_index+index]).to(self.device)
            road_edges = torch.from_numpy(self.gt['road_edges'][0][self.split_index+index]).to(self.device)
            leads = torch.from_numpy(self.gt['leads'][0][self.split_index+index]).to(self.device)
            leads_prob = torch.from_numpy(self.gt['leads_prob'][0][self.split_index+index]).to(self.device)
            desire_gt = torch.from_numpy(self.gt['desire'][0][self.split_index+index]).to(self.device)
            meta_various = torch.from_numpy(self.gt['meta_various'][0][self.split_index+index]).to(self.device)
            meta_desire = torch.from_numpy(self.gt['meta_desire'][0][self.split_index+index]).to(self.device)
            pose = torch.from_numpy(self.gt['pose'][0][self.split_index+index]).to(self.device)

            return (imgs, desire, traffic_conv, recurrent_state), (plan,
            ll, ll_prob, road_edges, leads, leads_prob, desire_gt, meta_various,
            meta_desire, pose)
            
class Transformations():
    
    def __init__(self):
    
        pass 

    def InitialImageTransform(self):
        
        pass 

    def ArrayToTensor(self):
        
        pass 


# if __name__ == "__main__":

    # numpy_paths = ["inputdata.npz","gtdata.npz"]
    # devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # comma_data = CommaLoader(numpy_paths,0.8, dummy_test= True, train= True)
    # comma_loader = DataLoader(comma_data, batch_size=2)

    # for i, data in enumerate(comma_loader):
    #     print(i)
