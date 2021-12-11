
import cv2
from sys import argv
import os
import glob 
import numpy as np
from tqdm import tqdm 
import h5py

path_file = argv[1]
gt_path_file = argv[2]

print("video_path:", path_file)
print("gt_path:", gt_path_file)

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
# for i in tqdm(range(len(hevc_file_paths))):
video_dir_path, hevc_file = os.path.split(path_file)
frame_file_path = video_dir_path + "/hevc_frames/"

if not os.path.exists(frame_file_path):
    os.makedirs(frame_file_path)
else :
    pass

loadhevc_frames(path_file,frame_file_path)
save_path_h5(gt_path_file)

print("Done")

