import sys
import numpy as np
from tqdm import tqdm
import h5py
import glob
from torch.utils.data import IterableDataset, DataLoader, Dataset
import os
import cv2
import math
import torch
import time
import threading
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa
from utils import bgr_to_yuv, transform_frames, printf  # noqa


MIN_SEGMENT_LENGTH = 1190

cache_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
path_to_videos_cache = os.path.join(cache_folder, 'videos.txt')
path_to_plans_cache = os.path.join(cache_folder, 'plans.txt')


class CommaDataset(IterableDataset):

    def __init__(self, recordings_basedir, train_split=0.8, seq_len=32, validation=False, shuffle=False, seed=42, single_frame_batches=False):
        super(CommaDataset, self).__init__()
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
        self.single_frame_batches = single_frame_batches

        if self.recordings_basedir is None or not os.path.exists(self.recordings_basedir):
            raise TypeError("recordings path is wrong")

        self.hevc_file_paths, self.gt_file_paths = self.get_paths(self.recordings_basedir, min_segment_len=MIN_SEGMENT_LENGTH)
        n_segments = len(self.hevc_file_paths)
        printf("Total # segments", n_segments)

        full_index = list(range(n_segments))

        # shuffle full train+val together *once*
        if self.shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(full_index)

        split_idx = int(np.round(n_segments * self.train_split))

        if not self.validation:
            self.segment_indices = full_index[:split_idx]
        else:
            self.segment_indices = full_index[split_idx:]

        printf("Subset # segments:", len(self.segment_indices))

    def __len__(self):
        return len(self.segment_indices)

    def __iter__(self):

        timing = dict()

        # shuffle data subset after each epoch
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.segment_indices)

        # support single & multi-processing
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_pid = os.getpid()
        cores_per_worker = os.cpu_count() // num_workers

        # force the process to be gentle, i.e. use no more than `cores_per_worker` cores instead of all
        
        ### uncomment the below line for making the dataloader more efficient.
        #subprocess.check_output(f'taskset -pc {worker_id*cores_per_worker}-{worker_id*cores_per_worker+cores_per_worker-1} {worker_pid}', shell=True)

        for segment_vidx in range(worker_id, len(self.segment_indices), num_workers):

            # retrieve true index of segment [0, 2331] using virtual index [0, 2331 * train_split]
            segment_idx = self.segment_indices[segment_vidx]

            segment_video = cv2.VideoCapture(self.hevc_file_paths[segment_idx])
            segment_gts = h5py.File(self.gt_file_paths[segment_idx], 'r')
            segment_length = segment_gts['plans'].shape[0]
            n_seqs = math.floor(segment_length / self.seq_len)

            _, frame2 = segment_video.read()  # initialize last frame
            yuv_frame2 = bgr_to_yuv(frame2)
            
            # TODO: (for further optimization) check if model can handle images pre-processed through (shorter) path2.
            # path1 and path2 look identical when saved to a PNG, but have some 
            # structured differences (to see, print the flattened difference between the two)

            # path1: bgr -> yuv -> rgb
            # path2: bgr -> rgb
            # path1 = yuv_to_rgb(bgr_to_yuv(frame2))
            # path2 = bgr_to_rgb(frame2)
            # printf('diff between paths:', list((path1 - path2).flatten()))

            for sequence_idx in range(n_seqs):

                segment_finished = sequence_idx == n_seqs-1

                if not self.single_frame_batches:
                    yuv_frame_seq = np.zeros((self.seq_len + 1, 1311, 1164), dtype=np.uint8)

                    yuv_frame_seq[0] = yuv_frame2


                # start iteration from 1 because we already read 1 frame before
                for t_idx in range(1, self.seq_len + 1):
                    sequence_finished = t_idx == self.seq_len

                    yuv_frame1 = yuv_frame2
                    _, frame2 = segment_video.read()

                    yuv_frame2 = bgr_to_yuv(frame2)

                    if self.single_frame_batches:
                        prepared_frames = transform_frames([yuv_frame1, yuv_frame2], timing)
                    else:
                        yuv_frame_seq[t_idx] = yuv_frame2

                    if self.single_frame_batches:
                        stacked_frames = np.vstack(prepared_frames).reshape(12, 128, 256)

                        abs_t_idx = sequence_idx*self.seq_len + t_idx
                        gt_plan = segment_gts['plans'][abs_t_idx]
                        gt_plan_prob = segment_gts['plans_prob'][abs_t_idx]

                        yield stacked_frames, gt_plan, gt_plan_prob, segment_finished, sequence_finished, worker_id

                if not self.single_frame_batches:
                    prepared_frames = transform_frames(yuv_frame_seq)

                    stacked_frame_seq = np.zeros((self.seq_len, 12, 128, 256), dtype=np.uint8)
                    for i in range(self.seq_len):
                        stacked_frame_seq[i] = np.vstack(prepared_frames[i:i+2])[None].reshape(12, 128, 256)

                    # shift slice by +1 to skip the 1st step which didn't see 2 stacked frames yet
                    abs_t_indices = slice(sequence_idx*self.seq_len+1, (sequence_idx+1)*self.seq_len+1)
                    gt_plan_seq = segment_gts['plans'][abs_t_indices]
                    gt_plan_prob_seq = segment_gts['plans_prob'][abs_t_indices]

                    yield stacked_frame_seq, gt_plan_seq, gt_plan_prob_seq, segment_finished, True, worker_id

            segment_gts.close()
            segment_video.release()

    def get_segment_dirs(self, base_dir):
        '''Get paths to segments that have ground truths.'''

        gt_files = sorted(glob.glob(base_dir + "/**/marker_and_leads_ground_truth.npz", recursive=True))
        return sorted(list(set([os.path.dirname(f) for f in gt_files])))

    def get_paths(self, base_dir, min_segment_len=1190):
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
            segment_dirs = self.get_segment_dirs(base_dir)

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


class BatchDataLoader:
    '''Assumes batch_size == num_workers to ensure same ordering of segments in each batch'''

    def __init__(self, loader, batch_size):
        self.loader = loader
        self.batch_size = batch_size

    def __iter__(self):
        bs = self.batch_size
        batch = [None] * bs
        current_bs = 0
        workers_seen = set()
        for d in self.loader:
            worker_id = d[-1]

            # this means there're fewer segments left than the size of the batch — drop the last ones
            if worker_id in workers_seen:
                printf(f'WARNING: sequence from worker:{worker_id} already seen in this batch. Dropping segments.')
                return  # FIXME: maybe pad the missing sequences with zeros and yield?

            batch[worker_id] = d
            current_bs += 1
            workers_seen.add(worker_id)

            if current_bs == bs:
                collated_batch = self.collate_fn(batch)
                yield collated_batch
                batch = [None] * bs
                current_bs = 0
                workers_seen = set()

    def collate_fn(self, batch):

        stacked_frames = torch.stack([item[0] for item in batch])
        gt_plan = torch.stack([item[1] for item in batch])
        gt_plan_prob = torch.stack([item[2] for item in batch])
        segment_finished = torch.tensor([item[3] for item in batch])
        sequence_finished = torch.tensor([item[4] for item in batch])

        return stacked_frames, gt_plan, gt_plan_prob, segment_finished, sequence_finished

    def __len__(self):
        return len(self.loader)


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator):
        threading.Thread.__init__(self)
        self.queue = torch.multiprocessing.Queue(2)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        while True:
            for item in self.generator:
                self.queue.put(item)
            self.queue.put(None)

    def __iter__(self):
        try:
            next_item = self.queue.get()
            while next_item is not None:
                yield next_item
                next_item = self.queue.get()
        except (ConnectionResetError, ConnectionRefusedError):
            self.stop()
            raise StopIteration
"""
Loader for visualization
"""

def prepare_frames(frame1, frame2):
    yuv_frame1 = bgr_to_yuv(frame1)
    yuv_frame2 = bgr_to_yuv(frame2)
    list_yuv_frame = [yuv_frame1, yuv_frame2]

    prepared_frames = transform_frames(list_yuv_frame)
    # print(prepared_frames[0].shape)
    stack_frames = np.zeros((1,12,128,256))
    stack_frames = (np.vstack((prepared_frames[0], prepared_frames[1]))).reshape(1,12,128,256)
    # print(stack_frames.shape)

    return stack_frames 

class viz_loader(Dataset):
    def __init__(self, video_path):
        super(viz_loader, self).__init__()
        
        self.video_path = video_path
        self.yuv_frames = []
        self.RGB_frames = []
        filelist = os.listdir(self.video_path)
        filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0])) 
        
        for i in range(len(filelist) -1):
            path_frame1 = os.path.join(self.video_path, filelist[i])
            path_frame2 = os.path.join(self.video_path,filelist[i+1])
            
            frame_1 =cv2.imread(path_frame1)
            frame_2 = cv2.imread(path_frame2)
#             print(frame_1.shape)
#             print(frame_2.shape)
            
            stacked_yuv_frames = prepare_frames(frame_1, frame_2)
            self.yuv_frames.append(stacked_yuv_frames)
            self.RGB_frames.append(frame_2)
#         print(len(self.yuv_frames))
    def __len__(self):
        return len(self.yuv_frames)

    def __getitem__(self, index):
        
        yuv_data = self.yuv_frames[index]
        yuv_data = torch.from_numpy(yuv_data).float()
        rgb_data  = self.RGB_frames[index]
        return yuv_data, rgb_data
    
if __name__ == "__main__":

    if len(sys.argv) >= 2:
        comma_recordings_basedir = sys.argv[1]
    else:
        comma_recordings_basedir = "/gpfs/space/projects/Bolt/comma_recordings"

    batch_size = num_workers = 30  # MUST BE batch_size == num_workers
    seq_len = 100
    prefetch_factor = 2
    prefetch_warmup_time = 2  # seconds wait before starting iterating
    train_split = 0.8
    single_frame_batches = False  # set True to serve batches of single frames instead of sequences

    assert batch_size == num_workers, 'Batch size must be equal to number of workers'

    # mock model
    simulated_model_fps = 500
    simulated_forward_time = batch_size * seq_len / simulated_model_fps

    # hack to get batches of different segments with many workers
    train_dataset = CommaDataset(comma_recordings_basedir, train_split=train_split, seq_len=seq_len, shuffle=True, single_frame_batches=single_frame_batches)
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers, shuffle=False, prefetch_factor=prefetch_factor, persistent_workers=True, collate_fn=None)
    train_loader = BatchDataLoader(train_loader, batch_size=batch_size)
    train_loader = BackgroundGenerator(train_loader)

    printf("Checking loader shapes...")
    printf()
    for epoch in range(3):
        for idx, batch in enumerate(train_loader):
            frames, plans, plans_probs, segment_finished, sequence_finished = batch

            printf(f'Frames: {frames.shape}. Plans: {plans.shape}. Plan probs: {plans_probs.shape}. Segment finished: {segment_finished.shape}. Sequence finished: {sequence_finished.shape}')

            if idx == 0:
                printf(f'Warming up pre-fetching {prefetch_warmup_time}s...')
                time.sleep(prefetch_warmup_time)  # keep this line in production for a few seconds to warm up the prefetching

            # <model goes here>
            time.sleep(simulated_forward_time)

