import numpy as np
import os
from tqdm.auto import tqdm
import glob

cache_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
path_to_videos_cache = os.path.join(cache_folder, 'videos.txt')
path_to_gts_cache = os.path.join(cache_folder, 'ground_truths.txt')


def get_segment_dirs(base_dir):
    gt_files = sorted(glob.glob(base_dir + "/**/marker_and_leads_ground_truth.npz", recursive=True))
    return sorted(list(set([os.path.dirname(f) for f in gt_files])))


def get_paths(base_dir):

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

if __name__ == '__main__':

	videos, gts = get_paths('/gpfs/space/projects/Bolt/comma_recordings')

	assert len(videos) == len(gts)

	print('Found segments:', len(videos))