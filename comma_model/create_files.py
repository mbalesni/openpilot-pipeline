
import cv2
from sys import argv
import os
import glob 
import numpy as np
from onnx import _get_file_path
from tqdm import tqdm 
import h5py
print("len sys arg",len(argv))
path_file = argv[1]
print("file_path:", path_file)

print("check if video file exists ", os.path.exists(path_file) )
### load npz files by iterate over gt_files and save .h5 file in the same path
def save_path_h5(file_path):
    path, file = os.path.split(file_path)
    data = np.load(file_path)
    path_data = data["plans"][:1190,:,:,:,:] ## clipping the data until 1190 dimensions only
    path_data_prob = data["plans_prob"][:1190,:,:]
    
    ## adding path plan and plan prob gt_data into .h5 file to make it iterable
    h5file_object  = h5py.File(path+ "/plan.h5",'w')
    h5file_object.create_dataset("plans",data = path_data)
    h5file_object.create_dataset("plans_prob", data = path_data_prob) 
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

print("saving .h5 files and converting the video file into frame and storing in hevc_frame folder")
video_dir_path, hevc_file = os.path.split(path_file)

print("check if the file exists", os.listdir(video_dir_path))
frame_file_path = video_dir_path + "/hevc_frames/"

gt_file_path = os.path.join(video_dir_path, "marker_and_leads_ground_truth.npz")

print("check if gt path exists",os.path.exists(gt_file_path))

if not os.path.exists(frame_file_path):
    os.makedirs(frame_file_path)
else :
    pass

loadhevc_frames(path_file,frame_file_path)
save_path_h5(gt_file_path)

print("Done")