# from comma_model.Dataloader import CommaLoader
# from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm.auto import tqdm

# path_comma_recordings = '/gpfs/space/projects/Bolt/comma_recordings'
# path_npz_dummy = ['inputdata.npz','gtdata.npz'] # dummy data_path
# onnx_path = 'supercombo.onnx'
# datatype = 'gen_gt'

# dataset = CommaLoader(path_comma_recordings, path_npz_dummy, 0.8, datatype, train=True)
# loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)

# print('length of loader:', len(loader))

# for batch in loader:
#   print(batch.shape)

hevc_file_paths = []
gt_file_paths = []

segment_dirs = []

with open('segments.txt', 'r') as f:
    segment_dirs = f.read().splitlines()

# prevent duplicate writes
with open('videos.txt', 'w') as video_paths: pass
with open('ground_truths.txt', 'w') as gt_paths: pass


gt_filename = 'marker_and_leads_ground_truth.npz'
video_filenames = ['fcamera.hevc', 'video.hevc']


# video_paths = open('videos.txt', 'a')
# gt_paths = open('ground_truths.txt', 'a')

for segment_dir in tqdm(segment_dirs):

	gt_data = np.load(os.path.join(segment_dir, gt_filename))

	if gt_data['plans'].shape[0] >= 1190:  # keep segments that have >= 1190 samples

		video_files = os.listdir(segment_dir)
		video_files = [file for file in video_files if file in video_filenames]

		found_one_video = 0 <= len(video_files) <= 1

		if found_one_video:
			with open('videos.txt', 'a') as video_paths:
				video_paths.write(os.path.join(segment_dir, video_files[0]) + '\n')

			with open('ground_truths.txt', 'a') as gt_paths:
				gt_paths.write(os.path.join(segment_dir, gt_filename) + '\n')
		else:
			print(f'WARNING: found {len(video_files)} in segment: {segment_dir}')
					

# video_paths.close()
# gt_paths.close()