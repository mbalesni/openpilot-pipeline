import os 
import sys

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


def parse_logs(path_segment, path_to_openpilot):
    """
    Create a function that will take the path and gives out the extrinsic and rpy values to be saved in .h5
    """
    sys.path.append(path_to_openpilot)
    from tools.lib.logreader import LogReader

    raw_log = os.path.join(path_segment, 'raw_log.bz2')
    r_log = os.path.join(path_segment, 'raw_log.bz2')
    if os.path.exists(raw_log):
        log_file = raw_log
    elif os.path.exists(r_log):
        log_file = r_log
    else:
        raise FileNotFoundError('Could not find raw_log.bz2 or rlog.bz2 in {}'.format(path_segment))

    frames_per_segment = 1200 # just for completeness; only 1190 are used for training
    
    msgs = LogReader(log_file)
    live_calibration_params = [m.liveCalibration for m in msgs if m.which() == 'liveCalibration']
    length_of_params  = len(live_calibration_params)
    rpy_segment = np.zeros((frames_per_segment, 3,1))
    extrinsic_matrix_segment = np.zeros((frames_per_segment, 3,4)) 
    
    repeat_n_times = int(np.round(frames_per_segment/length_of_params))

    count = 0

    for it,data in enumerate(live_calibration_params):
        
        if it == length_of_params-1:

            break
        ext_matrix = np.array(data.extrinsicMatrix).reshape(3,4)
        rotation_matrix = ext_matrix[:3,:3]
        
        cal_rpy_values = rotationMatrixToEulerAngles(rotation_matrix)
        rpy_calib = np.array(data.rpyCalib)[:, np.newaxis]

        if len(rpy_calib) == 0:
            count += 1
            rpy_calib = cal_rpy_values[:, np.newaxis]

        st_idx = repeat_n_times * it
        end_idx = repeat_n_times * (it+1)
        
        rep_ext = np.repeat(ext_matrix.reshape(1,3,4), repeat_n_times,  axis=0)        
        rep_rpy = np.repeat(rpy_calib.reshape(1,3,1), repeat_n_times, axis=0)

        extrinsic_matrix_segment[st_idx:end_idx, :,:] = rep_ext
        rpy_segment[st_idx:end_idx,:,:] = rep_rpy 
        
    return rpy_segment, extrinsic_matrix_segment


def save_segment_calib(segment_path, openpilot_dir, force=False):

    out_path = os.path.join(segment_path, 'calib.h5')

    if os.path.exists(out_path) and not force:
        print('Calibration already exists at:', out_path)
        return

    rpy_Calib, extrinsic_matrix = parse_logs(segment_path, openpilot_dir)

    with h5py.File(os.path.join(segment_path, "calib.h5"), 'w') as h5file_object:
        h5file_object.create_dataset("rpy", data=rpy_Calib)
        h5file_object.create_dataset("ext_matrix", data=extrinsic_matrix) 

if __name__ == '__main__':
    save_segment_calib('/home/nikita/data/2021-09-19--10-22-59/10', '/home/nikita/openpilot')