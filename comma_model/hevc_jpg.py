print("I am in the job")

import cv2
import sys 
import os
import glob 
import numpy as np
from tqdm import tqdm 
import h5py
sys.path.append("../")
from utils import get_train_imgs

comma_recordings_path = "/gpfs/space/projects/Bolt/comma_recordings"

gt_files_exist_path = sorted(glob.glob(comma_recordings_path + "/**/marker_and_leads_ground_truth.npz", recursive= True))

hevc_file_paths = []
gt_file_paths = []

print("loading the respective paths")
for paths in tqdm(gt_files_exist_path):
    dir_path, file_name = os.path.split(paths)
    check_data = np.load(paths)
    if check_data['plans'].shape[0] == 1190 or check_data['plans'].shape[0] > 1190: ## adding only those files who has more than 1190 samples
        # print(check_data['plans'].shape[0]
        if any(fname.endswith('.hevc') for fname in os.listdir(dir_path)):
            for file in os.listdir(dir_path):
                if file == "fcamera.hevc" or file == "video.hevc":
                    hevc_file_paths.append(os.path.join(dir_path,file))
                    gt_file_paths.append(os.path.join(dir_path,file_name))
print("paths loaded")
print(len(hevc_file_paths))
print(len(gt_file_paths))

### load npz files by iterate over gt_files and save .h5 file in the same path
def save_path_h5(file_path):
    path, file = os.path.split(file_path)
    data = np.load(file_path)
    path_data = data["plans"][:1190,:,:,:,:] ## clipping the data until 1190 dimensions only
    h5file_object  = h5py.File(path+ "/plan.h5",'w')
    h5file_object.create_dataset("plans",data = path_data) 
    h5file_object.close()

# load .hevc file and save the frames in the same path
def loadhevc_frames(video_path, folder_path):
    cap = cv2.VideoCapture(video_path)
    index = 0

    while cap.isOpened():
        ret, frame = cap.read()       
        if not ret:
            break
    
        cv2.imwrite(folder_path + str(index) + ".png", frame )
        index +=1
            
## loadhevc_frames(path, f_path)
print("saving .h5 files and converting the video file into frame and storing in hevc_frame folder")
for i in tqdm(range(len(hevc_file_paths))):
    video_dir_path, hevc_file = os.path.split(hevc_file_paths[i])
    frame_file_path = video_dir_path + "/hevc_frames/"

    if not os.path.exists(frame_file_path):
        os.makedirs(frame_file_path)
    else :
        pass

    loadhevc_frames(hevc_file_paths[i],frame_file_path)
    save_path_h5(gt_file_paths[i])

print("Done")

# # path = "/gpfs/space/projects/Bolt/comma_recordings/realdata/2021-05-26--08-50-16/0/fcamera.hevc"
# # f_path = "/gpfs/space/projects/Bolt/comma_recordings/realdata/2021-05-26--08-50-16/0/frames_hevc/"

# save_path_h5("/gpfs/space/projects/Bolt/comma_recordings/realdata/2021-05-26--08-50-16/0/marker_and_leads_ground_truth.npz")


    
        




