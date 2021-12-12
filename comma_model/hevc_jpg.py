print("I am in the job")
import sys 
import os
import glob 
import numpy as np
from tqdm import tqdm 

import subprocess
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

# path = "/gpfs/space/projects/Bolt/comma_recordings/realdata/2021-05-26--08-50-16/0/fcamera.hevc"
# f_path = "/gpfs/space/projects/Bolt/comma_recordings/realdata/2021-05-26--08-50-16/0/frames_hevc/"

# save_path_h5("/gpfs/space/projects/Bolt/comma_recordings/realdata/2021-05-26--08-50-16/0/marker_and_leads_ground_truth.npz")


single_job_file_path = os.path.join("/gpfs/space/home/gautamku/openpilot-pipeline/comma_model","single_job.sh" )
print("job file path:",single_job_file_path)

for i in range(len(hevc_file_paths)):
    os.system("sbatch %s %s" %(single_job_file_path, hevc_file_paths[i]))
# os.system("sbatch %s %s" %(single_job_file_path,"/gpfs/space/projects/Bolt/comma_recordings/realdata/2020-08-28--12-59-49--25/fcamera.hevc"))
