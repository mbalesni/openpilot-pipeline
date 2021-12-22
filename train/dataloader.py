import sys
import numpy as np
from tqdm import tqdm
import h5py
import glob
from torch.utils.data import IterableDataset, DataLoader
import os
import cv2
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa
from utils import bgr_to_yuv, transform_frames, printf  # noqa


MIN_SEGMENT_LENGTH = 1190

cache_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
path_to_videos_cache = os.path.join(cache_folder, 'videos.txt')
path_to_plans_cache = os.path.join(cache_folder, 'plans.txt')


class CommaLoader(IterableDataset):

    def __init__(self, recordings_basedir, train_split, seq_len=32, validation=False, shuffle=False, seed=42):
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
        self.validation = validation
        self.train_split = train_split
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.seed = seed

        if self.recordings_basedir is None or not os.path.exists(self.recordings_basedir):
            raise TypeError("recordings path is wrong")

        self.hevc_file_paths, self.gt_file_paths = get_paths(self.recordings_basedir, min_segment_len=MIN_SEGMENT_LENGTH)
        n_segments = len(self.hevc_file_paths)
        printf("# of segments", n_segments)

        full_index = list(range(n_segments))

        # shuffle full train+val together *once*
        if self.shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(full_index)

        split_idx = int(np.round(n_segments * self.train_split))

        if not self.validation:
            self.index = full_index[:split_idx]
        else:
            self.index = full_index[split_idx:]

        printf("segments ready to load:", len(self.index))

    def __len__(self):
        return len(self.index)

    def __iter__(self):

        # shuffle data subset after each epoch
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.index)

        for segment_idx in self.index:

            self.video = cv2.VideoCapture(self.hevc_file_paths[segment_idx])
            self.gts = h5py.File(self.gt_file_paths[segment_idx], 'r')
            self.segment_length = self.gts['plans'].shape[0]

            _, frame2 = self.video.read() # initialize last frame
            yuv_frame2 = bgr_to_yuv(frame2)

            n_seqs = math.floor(self.segment_length / self.seq_len)

            new_segment = True

            for sequence_idx in range(n_seqs):

                if sequence_idx > 0: new_segment = False

                stacked_frame_seq = np.zeros((self.seq_len, 12, 128, 256), dtype=np.uint8)
                
                # start iteration from 1
                # to skip the 1st plan step which didn't see 2 stacked frames yet 
                for t_idx in range(1, self.seq_len):
                    yuv_frame1 = yuv_frame2
                    _, frame2 = self.video.read()

                    yuv_frame2 = bgr_to_yuv(frame2)

                    prepared_frames = transform_frames([yuv_frame1, yuv_frame2])
                    stacked_frames = np.vstack(prepared_frames).reshape(1, 12, 128, 256)
                    stacked_frame_seq[t_idx] = stacked_frames

                abs_t_indices = slice(sequence_idx*self.seq_len, (sequence_idx+1)*self.seq_len)
                gt_plan_seq = self.gts['plans'][abs_t_indices]
                gt_plan_prob_seq = self.gts['plans_prob'][abs_t_indices]

                yield stacked_frame_seq, (gt_plan_seq, gt_plan_prob_seq), new_segment

            self.gts.close()

        return


def get_segment_dirs(base_dir):
    '''Get paths to segments that have ground truths.'''

    gt_files = sorted(glob.glob(base_dir + "/**/marker_and_leads_ground_truth.npz", recursive=True))
    return sorted(list(set([os.path.dirname(f) for f in gt_files])))


def get_paths(base_dir, min_segment_len=1190):
    '''Get paths to videos and ground truths. Cache them for future reuse.'''

    os.makedirs(cache_folder, exist_ok=True)

    if os.path.exists(path_to_videos_cache) and os.path.exists(path_to_plans_cache):
        printf('Using cached paths to videos and GTs...')
        video_paths = []
        gt_paths = []
        with open(path_to_videos_cache, 'r') as f:
            video_paths = f.read().splitlines()
        with open(path_to_plans_cache, 'r') as f:
            gt_paths = f.read().splitlines()
    else:
        printf('Resolving paths to videos and GTs...')
        segment_dirs = get_segment_dirs(base_dir)

        # prevent duplicate writes
        with open(path_to_videos_cache, 'w') as video_paths:
            pass
        with open(path_to_plans_cache, 'w') as gt_paths:
            pass

        gt_filename = 'plan.h5'
        video_filenames = ['fcamera.hevc', 'video.hevc']

        video_paths = []
        gt_paths = []

        for segment_dir in tqdm(segment_dirs):

            # TODO: wasn't tested after switch to `plan.h5` from npz files
            gt_file_path = os.path.join(segment_dir, gt_filename)
            if not os.path.exists(gt_file_path):
                printf(f'WARNING: not found plan.h5 file in segment: {segment_dir}')
                continue

            gt_plan = h5py.File(gt_file_path, 'r')['plans']

            if gt_plan.shape[0] >= min_segment_len:  # keep segments that have >= 1190 samples

                video_files = os.listdir(segment_dir)
                video_files = [file for file in video_files if file in video_filenames]

                found_one_video = 0 <= len(video_files) <= 1

                if found_one_video:
                    with open(path_to_videos_cache, 'a') as video_paths_f:
                        video_path = os.path.join(segment_dir, video_files[0])
                        video_paths.append(video_path)
                        video_paths_f.write(video_path + '\n')  # cache it

                    with open(path_to_plans_cache, 'a') as gt_paths_f:
                        gt_paths.append(gt_file_path)
                        gt_paths_f.write(gt_file_path + '\n')  # cache it
                else:
                    printf(f'WARNING: found {len(video_files)} in segment: {segment_dir}')

    return video_paths, gt_paths


if __name__ == "__main__":
    comma_recordings_basedir = "/gpfs/space/projects/Bolt/comma_recordings"

    train_dataset = CommaLoader(comma_recordings_basedir, 0.8, shuffle=False)
    valid_dataset = CommaLoader(comma_recordings_basedir, 0.8, shuffle=False, validation=True)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False)

    printf("checking the shapes of the loader outs")
    for idx, batch in enumerate(valid_loader):
        frames, (plan, plan_prob), is_new_segment = batch

        printf('Batch', idx)
        printf('frames:', frames.shape)
        printf('plan:', frames.shape)
        printf('plan_prob:', frames.shape)
        printf('is_new_segment:', frames.shape)
        printf()
        
        if idx > 32 * 37:
            break
