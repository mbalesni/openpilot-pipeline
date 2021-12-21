print("yes I am in the job ")
from operator import gt
import numpy as np 
from numpy import load
import torch 
import os
from torch._C import device 
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms
import sys 
sys.path.append("../")
from utils import *
import glob
import h5py
from tqdm import tqdm


cache_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
path_to_videos_cache = os.path.join(cache_folder, 'videos.txt')
path_to_gts_cache = os.path.join(cache_folder, 'ground_truths.txt')


class CommaLoader(Dataset):

    def __init__(self, recordings_basedir ,npz_paths, split_per, data_type, train= False, val = False):
        super(CommaLoader, self).__init__()
        """
        Dataloader for Comma model train. pipeline

        Summary: 
            This dataloader can be used for intial testing and for proper training
            Images are converted into YUV 4:2:0 channels and brought to a calib frame of reff 
            as used in the official comma pipeline.
        
        Args: ------------------
        """
        self.recordings_basedir = recordings_basedir
        self.data_type = data_type
        self.npz_paths = npz_paths 
        self.train = train
        self.val = val
        self.split_per = split_per

        if self.npz_paths is None:
            raise TypeError("add dummy data paths")

        elif self.recordings_basedir is None:
            raise TypeError("recordings path is wrong")

        elif self.data_type == "dummy":
            self.input = np.load(self.npz_paths[0])
            self.gt = np.load(self.npz_paths[1])

            numberofsamples = self.input['imgs'][0].shape
            length_data = numberofsamples[0]
            self.split_index = int(np.round(length_data * self.split_per))
            
            if self.train:
                len_for_loader = self.split_index 
                self.sample_length = len_for_loader   
            
            elif self.val:
                len_for_loader = length_data - self.split_index
                self.sample_length = len_for_loader

        elif self.data_type == "gen_gt":

            self.hevc_file_paths, self.gt_file_paths = get_paths(self.recordings_basedir)

            print("paths loaded")
            print("length of video files", len(self.hevc_file_paths))
            print("lenght of gt files", len(self.gt_file_paths))
            # print(self.hevc_file_paths)
            n_dirs = len(self.hevc_file_paths) 
            number_samples = 1190 ## 1190 sample in every dir
            self.split_index = int(np.round(n_dirs * self.split_per))         
            
            if self.train:
                self.gt_files = self.gt_file_paths[:self.split_index]
                self.hevc_files = self.hevc_file_paths[:self.split_index]  
                self.sample_length = self.split_index * number_samples
            
            elif self.val:
                val_len = n_dirs - self.split_index
                self.gt_files  =  self.gt_file_paths[self.split_index:]
                self.hevc_files = self.hevc_file_paths[self.split_index:]
                self.sample_length =  val_len * number_samples
        print("length of samples:", self.sample_length)
        
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
        
        h5gt_data = h5py.File(h5_file_fullpath, 'r') 
        gt_plan = h5gt_data['plans'][sample_index]
        gt_plan_prob = h5gt_data['plans_prob'][sample_index]

        gt_plan = np.array(gt_plan)
        gt_plan_prob = np.array(gt_plan_prob)
        h5gt_data.close()

        return stack_frames, gt_plan, gt_plan_prob 

    def __len__(self):
        return self.sample_length

    def __getitem__(self, index):
        
        if self.data_type == "dummy" and self.train:
            
            imgs = torch.from_numpy(self.input['imgs'][0][index]).float()
            desire = torch.from_numpy(self.input['desire'][0][index]).float()
            traffic_conv = torch.from_numpy(self.input['traff_conv'][0][index]).float()
            
            #need only during intialisation
            recurrent_state = torch.from_numpy(self.input['recurrent_state'][0][index]).float()
            
            plan = torch.from_numpy(self.gt['plan'][0][index]).float()
            plan_prob = torch.from_numpy(self.gt['plan_prob'][0][index]).float()
            ll = torch.from_numpy(self.gt['ll'][0][index]).float()
            ll_prob = torch.from_numpy(self.gt['ll_prob'][0][index]).float()
            road_edges = torch.from_numpy(self.gt['road_edges'][0][index]).float()
            leads = torch.from_numpy(self.gt['leads'][0][index]).float()
            leads_prob = torch.from_numpy(self.gt['leads_prob'][0][index]).float()
            lead_prob = torch.from_numpy(self.gt['lead_prob'][0][index]).float()
            desire_gt = torch.from_numpy(self.gt['desire'][0][index]).float()
            meta_eng = torch.from_numpy(self.gt['meta_eng'][0][index]).float()
            meta_various = torch.from_numpy(self.gt['meta_various'][0][index]).float()
            meta_blinkers = torch.from_numpy(self.gt['meta_blinkers'][0][index]).float()
            meta_desire = torch.from_numpy(self.gt['meta_desire'][0][index]).float()
            pose = torch.from_numpy(self.gt['pose'][0][index]).float()

            return (imgs, desire, traffic_conv, recurrent_state), (plan, plan_prob,
            ll, ll_prob, road_edges, leads, leads_prob, lead_prob, desire_gt, meta_eng, meta_various, meta_blinkers,
            meta_desire, pose)
        
        elif self.data_type == "gen_gt":
            dir_index = index // 1190
            sample_index = index % 1190
            
            datayuv, data_gt, data_gt_prob = self.populate_data(self.hevc_files, dir_index, sample_index)
            ## when use multiple workers put the data tensors on device in train loop
            datayuv = torch.from_numpy(datayuv)
            data_gt = torch.from_numpy(data_gt)
            data_gt_prob = torch.from_numpy(data_gt_prob)
            return datayuv, (data_gt, data_gt_prob)


def get_segment_dirs(base_dir):
    '''Get paths to segments that have ground truths.'''

    gt_files = sorted(glob.glob(base_dir + "/**/marker_and_leads_ground_truth.npz", recursive=True))
    return sorted(list(set([os.path.dirname(f) for f in gt_files])))


def get_paths(base_dir):
    '''Get paths to videos and ground truths. Cache them for future reuse.'''

	os.makedirs(cache_folder, exist_ok=True)

	if os.path.exists(path_to_videos_cache) and os.path.exists(path_to_gts_cache):
		print('Using cached paths to videos and GTs...')
		video_paths = []
		gt_paths = []
		with open(path_to_videos_cache, 'r') as f: video_paths = f.read().splitlines()
		with open(path_to_gts_cache, 'r') as f: gt_paths = f.read().splitlines()
	else:
		print('Resolving paths to videos and GTs...')
		segment_dirs = get_segment_dirs(base_dir)

		# prevent duplicate writes
		with open(path_to_videos_cache, 'w') as video_paths: pass
		with open(path_to_gts_cache, 'w') as gt_paths: pass

		gt_filename = 'marker_and_leads_ground_truth.npz'
		video_filenames = ['fcamera.hevc', 'video.hevc']

		video_paths = []
		gt_paths = []

		for segment_dir in tqdm(segment_dirs):

			gt_data = np.load(os.path.join(segment_dir, gt_filename))

			if gt_data['plans'].shape[0] >= 1190:  # keep segments that have >= 1190 samples

				video_files = os.listdir(segment_dir)
				video_files = [file for file in video_files if file in video_filenames]

				found_one_video = 0 <= len(video_files) <= 1

				if found_one_video:
					with open(path_to_videos_cache, 'a') as video_paths_f:
						video_path = os.path.join(segment_dir, video_files[0])
						video_paths.append(video_path)
						video_paths_f.write(video_path + '\n') # cache it

					with open(path_to_gts_cache, 'a') as gt_paths_f:
						gt_path = os.path.join(segment_dir, gt_filename)
						gt_paths.append(gt_path)
						gt_paths_f.write(gt_path + '\n') # cache it
				else:
					print(f'WARNING: found {len(video_files)} in segment: {segment_dir}')

	return video_paths, gt_paths


# if __name__ == "__main__":
#     comma_recordings_basedir = "/gpfs/space/projects/Bolt/comma_recordings"

#     numpy_paths = ["inputdata.npz","gtdata.npz"]
#     comma_data = CommaLoader(comma_recordings_basedir, numpy_paths, 0.8, "gen_gt",train= True)
#     comma_loader = DataLoader(comma_data, batch_size=2, num_workers=2, shuffle= False)
    
#     print("checking the shapes of the loader outs")
#     for i, j in enumerate(comma_loader):
#         print("I am in the loop of prints.")
#         yuv, data = j
#         print("Testing the shape of input tensor",yuv[0].shape)
#         print("Testing the shape of labels",data[0].shape)
#         print("Testing the shape of labels",data[1].shape)