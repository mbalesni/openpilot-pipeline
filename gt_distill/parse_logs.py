import os 
import sys

import codecs
codecs.register_error("strict", codecs.backslashreplace_errors)

import math 
import numpy as np 
import h5py

from utils import printf


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
    Extract extrinsic matrix and rpy values from raw logs.
    """
    sys.path.append(path_to_openpilot)
    frames_per_segment = 1200 # just for completeness; only 1190 are used for training
    
    try:
        from tools.lib.logreader import LogReader

        raw_log = os.path.join(path_segment, 'raw_log.bz2')
        r_log = os.path.join(path_segment, 'rlog.bz2')
        if os.path.exists(raw_log):
            log_file = raw_log
        elif os.path.exists(r_log):
            log_file = r_log
        else:
            raise FileNotFoundError('Could not find raw_log.bz2 or rlog.bz2 in {}'.format(path_segment))

        msgs = LogReader(log_file)
        live_calibration_params = [m.liveCalibration for m in msgs if m.which() == 'liveCalibration']
        num_calib_updates  = len(live_calibration_params)
        rpy_segment = np.zeros((frames_per_segment, 3,1))
        extrinsic_matrix_segment = np.zeros((frames_per_segment, 3,4)) 
        
        repeat_n_times_global = int(np.round(frames_per_segment/num_calib_updates))

        count = 0
        count_not_calibrated = 0

        for it,data in enumerate(live_calibration_params):
            
            if it == num_calib_updates-1:
                break

            if data.calStatus != 1:
                count_not_calibrated += 1
            
            ext_matrix = np.array(data.extrinsicMatrix).reshape(3,4)
            rotation_matrix = ext_matrix[:3,:3]
            
            cal_rpy_values = rotationMatrixToEulerAngles(rotation_matrix)
            rpy_calib = np.array(data.rpyCalib)[:, np.newaxis]

            if len(rpy_calib) == 0:
                count += 1
                rpy_calib = cal_rpy_values[:, np.newaxis]

            repeat_n_times = min(repeat_n_times_global, frames_per_segment - it)

            rep_ext = np.repeat(ext_matrix.reshape(1,3,4), repeat_n_times,  axis=0)        
            rep_rpy = np.repeat(rpy_calib.reshape(1,3,1), repeat_n_times, axis=0)

            st_idx = repeat_n_times_global * it
            end_idx = st_idx + repeat_n_times

            if end_idx > frames_per_segment-1:
                continue

            extrinsic_matrix_segment[st_idx:end_idx, :,:] = rep_ext
            rpy_segment[st_idx:end_idx,:,:] = rep_rpy 

        if count_not_calibrated > 0:
            printf('[WARNING] {}/{} live calibration parameters were not calibrated'.format(count_not_calibrated, num_calib_updates))

    except Exception as err:
        printf('[ERROR] Could not parse live calibration parameters from {}'.format(path_segment))
        printf(err)
        return None, None
        
    return rpy_segment, extrinsic_matrix_segment


def save_segment_calib(segment_path, openpilot_dir, force=False):

    out_path = os.path.join(segment_path, 'calib.h5')

    if os.path.exists(out_path) and not force:
        print('Calibration already exists at:', out_path)
        return

    rpy_Calib, extrinsic_matrix = parse_logs(segment_path, openpilot_dir)

    if rpy_Calib is None or extrinsic_matrix is None:
        return

    with h5py.File(os.path.join(segment_path, "calib.h5"), 'w') as h5file_object:
        h5file_object.create_dataset("rpy", data=rpy_Calib)
        h5file_object.create_dataset("ext_matrix", data=extrinsic_matrix) 

if __name__ == '__main__':
    # should be used in generate_gt.py.
    #
    # but here's example standalone usage
    save_segment_calib('/home/nikita/data/2021-09-19--10-22-59/10', '/home/nikita/openpilot')