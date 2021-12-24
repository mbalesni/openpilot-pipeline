import sys
import numpy as np
from tqdm import tqdm
import h5py
import glob
from torch.utils.data import IterableDataset, DataLoader
import os
import cv2
import math
import torch
import time
from timing import Timing
import matplotlib.pyplot as plt
import threading
import subprocess





sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa
from utils import bgr_to_yuv, transform_frames, printf  # noqa


MIN_SEGMENT_LENGTH = 1190

cache_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
path_to_videos_cache = os.path.join(cache_folder, 'videos.txt')
path_to_plans_cache = os.path.join(cache_folder, 'plans.txt')


class CommaLoader(IterableDataset):

    def __init__(self, recordings_basedir, train_split=0.8, seq_len=32, validation=False, shuffle=False, seed=42, single_frame_batches=False):
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
        self.single_frame_batches = single_frame_batches

        if self.recordings_basedir is None or not os.path.exists(self.recordings_basedir):
            raise TypeError("recordings path is wrong")

        self.hevc_file_paths, self.gt_file_paths = get_paths(self.recordings_basedir, min_segment_len=MIN_SEGMENT_LENGTH)
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
        subprocess.check_output(f'taskset -pc {worker_id*cores_per_worker}-{worker_id*cores_per_worker+cores_per_worker-1} {worker_pid}', shell=True)

        for segment_vidx in range(worker_id, len(self.segment_indices), num_workers):
        # for segment_vidx in range(worker_id, num_workers, num_workers):

            # retrieve true index of segment [0, 2331] using virtual index [0, 2331 * train_split]
            segment_idx = self.segment_indices[segment_vidx]

            segment_video = cv2.VideoCapture(self.hevc_file_paths[segment_idx])
            segment_gts = h5py.File(self.gt_file_paths[segment_idx], 'r')
            segment_length = segment_gts['plans'].shape[0]
            n_seqs = math.floor(segment_length / self.seq_len)

            _, frame2 = segment_video.read()  # initialize last frame
            yuv_frame2 = bgr_to_yuv(frame2)
            
            # TODO: check if model can handle images pre-processed through (shorter) path2.
            # path1 and path2 look identical when saved to a PNG, but have some 
            # structured differences (to see, print the flattened difference between the two)

            # path1: bgr -> yuv -> rgb
            # path2: bgr -> rgb
            # path1 = yuv_to_rgb(bgr_to_yuv(frame2))
            # path2 = bgr_to_rgb(frame2)
            # printf('diff between paths:', list((path1 - path2).flatten()))

            for sequence_idx in range(n_seqs):

                start_time = time.time()

                segment_finished = sequence_idx == n_seqs-1

                if not self.single_frame_batches:
                    yuv_frame_seq = np.zeros((self.seq_len, 1311, 1164), dtype=np.uint8)
                    yuv_frame_seq[0] = yuv_frame2


                # start iteration from 1 because we already read 1 frame before
                for t_idx in range(1, self.seq_len):  # FIXME: +1?
                    sequence_finished = t_idx == self.seq_len-1

                    yuv_frame1 = yuv_frame2
                    with Timing(timing, f'read_frame-{worker_id}'):
                        _, frame2 = segment_video.read()

                    with Timing(timing, f'yuv_convert-{worker_id}'):
                        yuv_frame2 = bgr_to_yuv(frame2)

                    with Timing(timing, f'transform_frames_total-{worker_id}'):
                        if self.single_frame_batches:
                            prepared_frames = transform_frames([yuv_frame1, yuv_frame2], timing)
                        else: 
                            yuv_frame_seq[t_idx] = yuv_frame2

                    if self.single_frame_batches:
                        with Timing(timing, f'stack_frames-{worker_id}'):
                            stacked_frames = np.vstack(prepared_frames).reshape(12, 128, 256)

                        abs_t_idx = sequence_idx*self.seq_len + t_idx
                        gt_plan = segment_gts['plans'][abs_t_idx]
                        gt_plan_prob = segment_gts['plans_prob'][abs_t_idx]

                        yield stacked_frames, gt_plan, gt_plan_prob, segment_finished, sequence_finished, worker_id

                if not self.single_frame_batches:
                    with Timing(timing, f'transform_frames_total-{worker_id}'):
                        prepared_frames = transform_frames(yuv_frame_seq, timing)
                        # printf('prepared_frames', prepared_frames.shape)

                    with Timing(timing, f'stack_frames-{worker_id}'):
                        stacked_frame_seq = np.zeros((self.seq_len-1, 12, 128, 256), dtype=np.uint8)
                        for i in range(self.seq_len-1):
                            stacked_frame_seq[i] = np.vstack(prepared_frames[i:i+2])[None].reshape(12, 128, 256)

                    # shift slice by +1 to skip the 1st step which didn't see 2 stacked frames yet
                    abs_t_indices = slice(sequence_idx*self.seq_len+1, (sequence_idx+1)*self.seq_len+1)
                    gt_plan_seq = segment_gts['plans'][abs_t_indices]
                    gt_plan_prob_seq = segment_gts['plans_prob'][abs_t_indices]

                    delta_time = time.time() - start_time

                    if 'sequence_total' not in timing:
                        timing['sequence_total'] = {'time': 0, 'count': 0}
                    timing['sequence_total']['time'] += delta_time
                    timing['sequence_total']['count'] += self.seq_len

                    # yield stacked_frame_seq, gt_plan_seq, gt_plan_prob_seq, segment_finished, True, worker_id

                    # printf(f'worker: {worker_id}, segment: {segment_idx}, seq: {sequence_idx}, time: {time.time() - start_time:.2f}s')
                    yield segment_idx, sequence_idx, segment_finished, True, True, worker_id

            segment_gts.close()

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

    # TODO: test this running through the full dataset
    def __iter__(self):
        bs = self.batch_size
        batch = [None] * bs
        current_bs = 0
        workers_seen = set()
        for d in self.loader:
            worker_id = d[-1]

            # this means there're fewer segments left than the size of the batch — drop the last ones
            if worker_id in workers_seen:
                printf(f'Error: sequence from worker:{worker_id} already seen in this batch, dropping batch. Either a worker was too fast or there\'re fewer segments left than the batch size.')
                return  # FIXME: think about how to handle this case

            batch[worker_id] = d
            current_bs += 1
            workers_seen.add(worker_id)

            if current_bs == bs:
                collated_batch = self.collate_fn(batch)
                yield collated_batch
                batch = [None] * bs
                current_bs = 0
                workers_seen = set()

        # drop non-full batches
        return

    def collate_fn(self, batch):

        stacked_frames = torch.tensor([item[0] for item in batch])
        gt_plan = torch.tensor([item[1] for item in batch])
        gt_plan_prob = torch.tensor([item[2] for item in batch])
        segment_finished = torch.tensor([item[3] for item in batch])
        sequence_finished = torch.tensor([item[4] for item in batch])

        return stacked_frames, gt_plan, gt_plan_prob, segment_finished, sequence_finished

    def __len__(self):
        return len(self.loader)


def merge_timings(arr):
	"""
	Merge timings from different runs.

	Args:
		arr (list): list of dicts with timings

	Returns:
		dict: merged timings
	"""
	merged = {}
	for item in arr:
		for key, value in item.items():
			if key not in merged:
				merged[key] = {'time': 0, 'count': 0}
			merged[key]['time'] += value['time']
			merged[key]['count'] += value['count']
	return merged

def pprint_timing(merged_timings, batch_size):

	for key, value in merged_timings.items():
		printf('{}: {:.2f}s'.format(key, value['time'] / value['count']))


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator):
        threading.Thread.__init__(self)
        self.queue = torch.multiprocessing.Queue(2)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
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


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        comma_recordings_basedir = sys.argv[1]
    else:
        comma_recordings_basedir = "/gpfs/space/projects/Bolt/comma_recordings"


    model_fps_s = [400, 800, 1600]
    # model_fps_s = [400, 600]

    log_template = {
        'total_fps': [],
        'total_fps_error': [],
        'worker_fps': [],
        'batch_size': [],
    }
    logs = {k: {
        'total_fps': [],
        'total_fps_error': [],
        'worker_fps': [],
        'batch_size': [],
    } for k in model_fps_s}


        
    for simulated_model_fps in model_fps_s:
        # for batch_size in [8, 16]:
        for batch_size in [8, 16, 30]:

            num_workers = batch_size

            printf()
            printf(f'Test with {num_workers} workers | batch size: {batch_size}')

            seq_len = 100
            prefetch_factor = 2
            prefetch_warmup_time = 2  # wait before starting iterating
            simulated_forward_time = batch_size * seq_len / simulated_model_fps
            printf(f'Simulated model FPS: {simulated_model_fps} ({simulated_forward_time:.2f}s)')
            train_split = 0.8
            single_frame_batches = False

            # NOTE: num_workers corresponds to number of segments we are sampling from
            #
            # so if we have num_workers=1, we are sampling from 1 segment at a time
            # if num_workers < batch_size, a batch will have several sequences from the same segments
            #   e.g. batch = [A1, B1, C1, A2, B2], num_workers=3, batch_size=5; A2 means sequence 2 from segment A
            # if num_workers > batch_size, consecutive batches will have sequences from different segments
            #   e.g. batch1 = [A1, B1, C1], batch2 = [D1, E1, A2], num_workers=5, batch_size=3;
            #
            # !!! so need keep num_workers == batch_size !!!
            #
            assert num_workers == batch_size, 'num_workers must be equal to batch_size, see comment above'


            # hack to get batches of different segments with many workers
            train_dataset = CommaLoader(comma_recordings_basedir, train_split=train_split, seq_len=seq_len, shuffle=True, single_frame_batches=single_frame_batches)
            train_loader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers, shuffle=False, prefetch_factor=prefetch_factor, collate_fn=None)
            train_loader = BatchDataLoader(train_loader, batch_size=batch_size)
            train_loader = BackgroundGenerator(train_loader)

            timings = []
            fps_history = []
            fetch_delays = []

            printf("Starting loop (first batch will be slow)")
            printf()
            prev_time = time.time()
            for idx, batch in enumerate(train_loader):
                # frames, plans, plans_probs, segment_finished, sequence_finished = batch
                segment, sequence, segment_finished, sequence_finished, _ = batch

                time_delta = time.time() - prev_time
                printf(f'{time_delta:.3f}s - {batch_size * seq_len} frames (FPS: {batch_size * seq_len // time_delta}). Segments: {torch.unique(segment)}. Sequence: {torch.unique(sequence)}. Segment finished: {torch.unique(segment_finished)}. Sequence finished: {torch.unique(sequence_finished)}')
                # printf(f'{time_delta:.3f}s - {batch_size * seq_len} frames (FPS: {batch_size * seq_len // time_delta}). Frames: {frames.shape}. Plans: {plans.shape}. Plan probs: {plans_probs.shape}. Segment finished: {segment_finished.shape}. Sequence finished: {sequence_finished.shape}')

                if idx == 0:
                    printf(f'Warming up pre-fetching {prefetch_warmup_time}s...')
                    time.sleep(prefetch_warmup_time)

                fps = batch_size * seq_len / time_delta
                fps_history.append(fps)

                # fake foward pass
                time.sleep(simulated_forward_time)

                prev_time = time.time()

            # skip first batch, it's always slow (probably due to shuffling)
            total_fps = np.mean(fps_history[1:])
            total_fps_error = np.std(fps_history[1:])

            print(f'Append to model_fps:{simulated_model_fps} total_fps:{total_fps} batch_size:{batch_size}')
            logs[simulated_model_fps]['total_fps'].append(total_fps)
            logs[simulated_model_fps]['total_fps_error'].append(total_fps_error)
            logs[simulated_model_fps]['batch_size'].append(batch_size)

    # setup a wide figure
    fig, ax1 = plt.subplots(figsize=(16, 10))

    fig.suptitle('Data Loader Throughput', fontsize=16)
    ax1.set_title('Sequence length = 100', fontsize=12)
    ax1.set_xlabel('Batch size = Num workers')
    ax1.set_ylabel('FPS (frames per second)')
    ax1.set_xticks(logs[list(logs.keys())[0]]['batch_size'])

    for simulated_model_fps, log in logs.items():
        print('printing sim fps:', simulated_model_fps)
        stds_total_fps = np.array(log['total_fps_error'])

        print('batches:', log['batch_size'])
        print('total_fps:', log['total_fps'])

        ax1.plot(log['batch_size'], log['total_fps'], linestyle='--', marker='o', markersize=10, label=f'Total throughpout @ simulated model FPS: {simulated_model_fps}')
        ax1.fill_between(log['batch_size'], log['total_fps'] - stds_total_fps, log['total_fps'] + stds_total_fps, alpha=0.2)

    plt.legend()
    plt.savefig('FPS vs batch_size.png')

