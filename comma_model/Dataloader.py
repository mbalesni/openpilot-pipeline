print("yes I am in the job ")
from operator import gt
import numpy as np 
from numpy import load
import torch 
import os 
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms
import sys 
sys.path.append("../")
from utils import *
import glob
import h5py
from tqdm import tqdm

class CommaLoader(Dataset):

    def __init__(self, recordings_path ,npz_paths, split_per, data_type, transform = None, device = None, train= False, test = False):
        super(CommaLoader, self).__init__()
        """
        Dataloader for Comma model train. pipeline

        Summary: 
            This dataloader can be used for intial testing and for proper training
            Images are converted into YUV 4:2:0 channels and brought to a calib frame of reff 
            as used in the official comma pipeline.
        
        Args: ------------------
        """
        self.recordings_path = recordings_path
        self.data_type = data_type
        self.npz_paths = npz_paths 
        self.transform = transform
        self.device = device 
        self.train = train
        self.test = test
        self.split_per = split_per

        if self.npz_paths is None:
            raise TypeError("add dummy data paths")

        elif self.recordings_path is None:
            raise TypeError("recordings path is wrong")

        elif self.data_type == "dummy":
            self.input = np.load(self.npz_paths[0])
            self.gt = np.load(self.npz_paths[1])

        elif self.data_type == "gen_gt":
            gt_files_exist_path = sorted(glob.glob(self.recordings_path + "/**/marker_and_leads_ground_truth.npz", recursive= True))

            self.hevc_file_paths = []
            self.gt_file_paths = []
            print("loading the respective paths")
            for paths in tqdm(gt_files_exist_path):
                dir_path, file_name = os.path.split(paths)
                check_data = np.load(paths)
                if check_data['plans'].shape[0] == 1190 or check_data['plans'].shape[0] > 1190: ## adding only those files who has more than 1190 samples
                    # print(check_data['plans'].shape[0]
                    if any(fname.endswith('.hevc') for fname in os.listdir(dir_path)):
                        for file in os.listdir(dir_path):
                            if file == "fcamera.hevc" or file == "video.hevc":
                                self.hevc_file_paths.append(os.path.join(dir_path,file))
                                self.gt_file_paths.append(os.path.join(dir_path,file_name))
            print("paths loaded")
            
            self.n_dirs = len(self.hevc_file_paths) 
            self.number_samples = 1190 ## 1190 sample in every dir

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
    
    def populate_data(self, hevc_file_paths, dir_index, sample_index):
    
        path, file =os.path.split(hevc_file_paths[dir_index])
        frame_dir_name = "hevc_frames"
        h5_dir_name = "plan.h5"
        frame_dir_full_path = os.path.join(path, frame_dir_name)
        h5_file_fullpath = os.path.join(path,h5_dir_name)

        ### yuv stacked image of (sample_index) and (sample_index +1)
        # print(frame_dir_full_path + "/" + str(sample_index) + ".png")
        # print(frame_dir_full_path + "/" + str(sample_index+1) + ".png")
        frame_1 = cv2.imread(frame_dir_full_path + "/" +  str(sample_index) +".png")
        frame_2 = cv2.imread(frame_dir_full_path + "/" + str(sample_index+1) +".png")

        # print(frame_1.shape)
        # print(frame_2.shape)
        yuv_frame1 = bgr_to_yuv(frame_1)
        yuv_frame2 = bgr_to_yuv(frame_2)
        list_yuv_frame = [yuv_frame1, yuv_frame2]

        prepared_frames = transform_frames(list_yuv_frame)
        # print(prepared_frames[0].shape)
        stack_frames = np.zeros((1,12,128,256))
        stack_frames = (np.vstack((prepared_frames[0], prepared_frames[1]))).reshape(1,12,128,256)
        # print(stack_frames.shape)
        
        with h5py.File(h5_file_fullpath, 'r') as h5gt_data:
            gt_plan = h5gt_data['plans'][sample_index]
    
        return stack_frames, gt_plan 

    def __len__(self):

        if self.data_type =="dummy":
            sample_length = self.split_data(self.input)
            return sample_length
        
        elif self.data_type == "gen_gt":
            return self.n_dirs * self.number_samples 


    def __getitem__(self, index):
        
        if self.data_type == "dummy" and self.train:
            
            imgs = torch.from_numpy(self.input['imgs'][0][index]).float().to(self.device)
            desire = torch.from_numpy(self.input['desire'][0][index]).float().to(self.device)
            traffic_conv = torch.from_numpy(self.input['traff_conv'][0][index]).float().to(self.device)
            
            #need only during intialisation
            recurrent_state = torch.from_numpy(self.input['recurrent_state'][0][index]).float().to(self.device)
            
            plan = torch.from_numpy(self.gt['plan'][0][index]).float().to(self.device)
            plan_prob = torch.from_numpy(self.gt['plan_prob'][0][index]).float().to(self.device)
            ll = torch.from_numpy(self.gt['ll'][0][index]).float().to(self.device)
            ll_prob = torch.from_numpy(self.gt['ll_prob'][0][index]).float().to(self.device)
            road_edges = torch.from_numpy(self.gt['road_edges'][0][index]).float().to(self.device)
            leads = torch.from_numpy(self.gt['leads'][0][index]).float().to(self.device)
            leads_prob = torch.from_numpy(self.gt['leads_prob'][0][index]).float().to(self.device)
            lead_prob = torch.from_numpy(self.gt['lead_prob'][0][index]).float().to(self.device)
            desire_gt = torch.from_numpy(self.gt['desire'][0][index]).float().to(self.device)
            meta_eng = torch.from_numpy(self.gt['meta_eng'][0][index]).float().to(self.device)
            meta_various = torch.from_numpy(self.gt['meta_various'][0][index]).float().to(self.device)
            meta_blinkers = torch.from_numpy(self.gt['meta_blinkers'][0][index]).float().to(self.device)
            meta_desire = torch.from_numpy(self.gt['meta_desire'][0][index]).float().to(self.device)
            pose = torch.from_numpy(self.gt['pose'][0][index]).float().to(self.device)

            return (imgs, desire, traffic_conv, recurrent_state), (plan, plan_prob,
            ll, ll_prob, road_edges, leads, leads_prob, lead_prob, desire_gt, meta_eng, meta_various, meta_blinkers,
            meta_desire, pose)
        
        elif self.data_type == "gen_gt":

            dir_index = index // 1190
            sample_index = index % 1190

            datayuv, data_gt = self.populate_data(self.hevc_file_paths, dir_index, sample_index)
            datayuv = torch.from_numpy(datayuv).float.to(self.device)
            data_gt = torch.from_numpy(data_gt).float.to(self.device)

            return datayuv, data_gt
# if __name__ == "__main__":

#     numpy_paths = ["inputdata.npz","gtdata.npz"]
#     devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     comma_data = CommaLoader(numpy_paths,0.8, dummy_test= True, train= True)
#     comma_loader = DataLoader(comma_data, batch_size=2)

#     for i, j in comma_loader:
#         print(j[9].shape)

# comma_recordings_path = "/gpfs/space/projects/Bolt/comma_recordings"

# gt_files_exist_path = sorted(glob.glob(comma_recordings_path + "/**/marker_and_leads_ground_truth.npz", recursive= True))

# hevc_file_paths = []
# gt_file_paths = []

# print("loading the respective paths")
# for paths in tqdm(gt_files_exist_path):
#     dir_path, file_name = os.path.split(paths)
#     check_data = np.load(paths)
#     if check_data['plans'].shape[0] == 1190 or check_data['plans'].shape[0] > 1190: ## adding only those files who has more than 1190 samples
#         # print(check_data['plans'].shape[0]
#         if any(fname.endswith('.hevc') for fname in os.listdir(dir_path)):
#             for file in os.listdir(dir_path):
#                 if file == "fcamera.hevc" or file == "video.hevc":
#                     hevc_file_paths.append(os.path.join(dir_path,file))
#                     gt_file_paths.append(os.path.join(dir_path,file_name))
# print("paths loaded")

# def load_frames(frames_path):
#     yuv_frames = []
#     files_list = os.listdir(frames_path)
#     files_list = sorted(files_list, key=lambda x: int(os.path.splitext(x)[0]))
#     for files in files_list:
#         # print(os.path.join(frames_path, files))
#         frame = cv2.imread(os.path.join(frames_path, files))
#         yuv_frames.append(bgr_to_yuv(frame))
    
#     return yuv_frames
