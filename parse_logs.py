import os 
import glob 
import sys

sys.path.append('/home/nikita/openpilot/')
from tools.lib.logreader import LogReader
import codecs
codecs.register_error("strict", codecs.backslashreplace_errors)

import math 
import numpy as np 
import h5py

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])



def parse_logs(file_path):
    """
    Create a function that will take the path and gives out the extrinsic and rpy values to be saved in .h5
    """
    path_segment, f_path = os.path.split(file_path)
    frame_path = os.path.join(path_segment,"hevc_frames")

    no_frames_segment = 1200 # anyways we are just taking 1190 for training. 
    # print(no_frames_segment)
    
    msgs = LogReader(file_path)
    live_calibration_params = [m.liveCalibration for m in msgs if m.which() == 'liveCalibration']
    length_of_params  = len(live_calibration_params)
    # print(length_of_params)
    rpy_segment = np.zeros((no_frames_segment, 3,1))
    extrinsic_matrix_segment =np.zeros((no_frames_segment, 3,4)) 
    
    mul = int(np.round(no_frames_segment/length_of_params))
    # print(mul)

    for it,data in enumerate(live_calibration_params):
        
        if it == length_of_params-1:

            break
        ext_matrix = np.array(data.extrinsicMatrix).reshape(3,4)
        rotation_matrix = ext_matrix[:3,:3]
        
        cal_rpy_values = rotationMatrixToEulerAngles(rotation_matrix)
        rpy_calib = np.array(data.rpyCalib)[:, np.newaxis]

        if len(rpy_calib) ==0:
            print("&&")
            rpy_calib = cal_rpy_values[:, np.newaxis]

        st_idx = mul*it
        end_idx = mul*(it+1)
        
        rep_ext = np.repeat(ext_matrix.reshape(1,3,4),mul, axis =0)
        
        extrinsic_matrix_segment[st_idx:end_idx, :,:] = rep_ext
        
        rep_rpy = np.repeat(rpy_calib.reshape(1,3,1),mul,axis = 0)
        rpy_segment[st_idx:end_idx,:,:] = rep_rpy 
        
        return rpy_segment, extrinsic_matrix_segment


def create_log_file(file_path):

    path, _ = os.path.split(file_path) 
    # print(path)
    rpy_Calib, extrinsic_matrix = parse_logs(file_path)
    h5file_object  = h5py.File(path+ "/parsed_rlogs.h5",'w')
    h5file_object.create_dataset("rpy",data = rpy_Calib)
    h5file_object.create_dataset("ext_matrix", data = extrinsic_matrix) 
    h5file_object.close()

def get_log_paths(mount_path, cached_file_path):
    
    print("===> processing neuron paths from cached file")
    neuron_paths = []
    with open(cached_file_path) as f:
        file_paths = f.readlines()

        for i in range(len(file_paths)):
            file_path , _ = os.path.split(file_paths[i])
            
            neuron_path_split = file_path.split(os.sep)[6:]
            neuron_path_split = os.path.join(*neuron_path_split)
            neuron_path = os.path.join(mount_path, neuron_path_split)
            neuron_paths.append(neuron_path)
        
    print("===> processing raw log file paths")
    log_file_paths = []
    for i in range(len(neuron_paths)):
        for file in os.listdir(neuron_paths[i]):
            if file == 'raw_log.bz2' or file == 'rlog.bz2':
                log_file_path = os.path.join(neuron_paths[i],file)
                log_file_paths.append(log_file_path)

    return log_file_paths



mnt_path = '/mnt/openpilot_data'
cached_file_pth = '/home/gautam/plans.txt' 

paths = get_log_paths(mnt_path,cached_file_pth)

for i in range(len(paths)):
    create_log_file(paths[i])