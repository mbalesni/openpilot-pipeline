#!/usr/bin/env python3

from os.path import exists
import onnxruntime as ort
from tqdm import tqdm
import numpy as np
import sys
import os
import time
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import extract_preds, get_segment_dirs, printf
from train.dataloader import load_transformed_video


def frames_to_tensor(frames):
    H = (frames.shape[1]*2)//3
    W = frames.shape[2]
    in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

    in_img1[:, 0] = frames[:, 0:H:2, 0::2]
    in_img1[:, 1] = frames[:, 1:H:2, 0::2]
    in_img1[:, 2] = frames[:, 0:H:2, 1::2]
    in_img1[:, 3] = frames[:, 1:H:2, 1::2]
    in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))
    in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))
    return in_img1


def generate_ground_truth(path_to_segment, model, force=False):
    '''Model expected to be an onnxruntime InferenceSession.'''

    out_path = os.path.join(path_to_segment, 'gt_hacky.h5')

    if exists(out_path) and not force:
        print('Ground truth already exists at:', out_path)
        return

    input_frames, _ = load_transformed_video(path_to_segment)
    if input_frames is None: return

    input_frames = input_frames.numpy()
    recurrent_state = np.zeros((1, 512)).astype(np.float32)
    desire = np.zeros((1, 8)).astype(np.float32)
    tc = np.array([[0, 1]]).astype(np.float32)

    plans = []
    plans_prob = []
    lanelines = []
    laneline_probs = []
    road_edges = []
    road_edge_stds = []

    for img in input_frames:
        img = np.expand_dims(img.astype(np.float32), axis=0)
        outs = model.run(None, {'input_imgs': img, 'desire': desire, 'traffic_convention': tc, 'initial_state': recurrent_state})[0]

        results = extract_preds(outs, best_plan_only=False)[0]

        (lane_lines_t, lane_lines_probs_t), (road_edges_t, road_edges_std_t), (plans_t, plans_prob_t) = results

        plans.append(plans_t)
        plans_prob.append(plans_prob_t)

        lanelines.append(np.stack(lane_lines_t))
        laneline_probs.append(np.stack(lane_lines_probs_t))

        road_edges.append(np.stack(road_edges_t))
        road_edge_stds.append(np.stack(road_edges_std_t))

        # Important to refeed the state
        recurrent_state = outs[:, -512:]

    if not plans:
        return

    try:
        # delete existing file
        try:
            os.remove(out_path)
        except FileNotFoundError:
            pass

        with h5py.File(out_path, 'w') as h5file_object:
            h5file_object.create_dataset("plans", data=np.stack(plans))
            h5file_object.create_dataset("plans_prob", data=np.stack(plans_prob)) 
            h5file_object.create_dataset("lanelines", data=np.stack(lanelines))
            h5file_object.create_dataset("laneline_probs", data=np.stack(laneline_probs))
            h5file_object.create_dataset("road_edges", data=np.stack(road_edges))
            h5file_object.create_dataset("road_edge_stds", data=np.stack(road_edge_stds))
    except Exception as e:
        print(f'Couldn\'t save the ground truths at {path_to_segment}:', e)


if __name__ == '__main__':
    data_dir = sys.argv[1]

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_model = os.path.join(parent_dir, 'common/models/supercombo.onnx')

    options = ort.SessionOptions()
    options.intra_op_num_threads = 30
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # CPU turns out faster than CUDA with batch size = 1
    model = ort.InferenceSession(path_to_model, providers=["CPUExecutionProvider"], sess_options=options)

    printf('Looking for segments...')
    start_time = time.time()
    segment_dirs = get_segment_dirs(data_dir)

    # shuffle to allow multiple concurrent runs
    np.random.shuffle(segment_dirs)

    printf(f'\nFound a total of {len(segment_dirs)} segments. {time.time() - start_time:.2f}s \n')

    pbar = tqdm(segment_dirs, desc='Total progress:')
    for path_to_segment in pbar:
        start_time = time.time()
        generate_ground_truth(path_to_segment, model, force=False)
        printf(f'{time.time() - start_time:.2f}s - Done segment: {path_to_segment} ')

