#!/usr/bin/env python3

from os.path import exists
import onnxruntime as ort
from tqdm import tqdm
import numpy as np
import sys
import os
import time
import h5py
from pathlib import Path
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import extract_preds, printf, dir_path
from train.dataloader import load_transformed_video
from train.parse_logs import save_segment_calib


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
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(parent_dir, 'common/models/supercombo.onnx')

    parser = argparse.ArgumentParser(description='Run the original supercombo model on the dataset and save the predicted path plans.')
    parser.add_argument("--recordings_basedir", type=dir_path, default="/gpfs/space/projects/Bolt/comma_recordings", help="path to base directory with recordings")
    parser.add_argument("--openpilot_dir", type=dir_path, default=str(Path.home() / 'openpilot'), help="path to openpilot directory")
    parser.add_argument("--path_to_model", default=default_model_path, help="path to model for creating ground truths")
    parser.add_argument("--force_gt", dest='force_gt', action='store_true', help="path to model for creating ground truths")
    parser.add_argument("--force_calib", dest='force_calib', action='store_true', help="path to model for creating ground truths")
    parser.set_defaults(force_gt=False, force_calib=False)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 30
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # CPU turns out faster than CUDA with batch size = 1
    model = ort.InferenceSession(args.path_to_model, providers=["CPUExecutionProvider"], sess_options=options)

    # recursive os walk starting at recordings_basedir
    for dir_path, _, files in tqdm(os.walk(args.recordings_basedir)):
        if 'video.hevc' not in files and 'fcamera.hevc' not in files:
            continue

        generate_ground_truth(dir_path, model, force=args.force_gt)

        # TODO: test that this works & uncomment
        save_segment_calib(dir_path, args.openpilot_dir, force=args.force_calib)

        printf()
