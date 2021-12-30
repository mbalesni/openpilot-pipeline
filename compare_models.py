from onnx2pytorch import ConvertModel
from tensorflow.keras.layers import BatchNormalization
import time
import numpy as np
import torch
import onnxruntime as rt
import onnx
from onnx2keras import onnx_to_keras
from keract import get_activations
import colorama
import matplotlib.pyplot as plt
from train.dataloader import CommaDataset, BatchDataLoader, BackgroundGenerator
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
import os


def printf(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def color_thresh(x, threshold=1e-3):
    c = colorama.Fore.RED if x >= threshold else colorama.Fore.WHITE
    return f'{c}{x}'


def seperate_points_and_std_values(path):
    points_indices = np.arange(0, len(path), 2)
    std_indices = np.arange(1, len(path), 2)

    points = path[points_indices]
    std_values = path[std_indices]

    return points, std_values


PATH = 4955
LANE_LINES = PATH+528
LANE_LINE_PROB = LANE_LINES+8
ROAD_EDGES = LANE_LINE_PROB+264
LEADS = ROAD_EDGES+102
LEAD_PROB = LEADS+3
DESIRE_STATE = LEAD_PROB+8
META = DESIRE_STATE+80
POSE = META+12
RECURRENT_STATE = POSE+512

TRAJECTORY_SIZE = 33
MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

X_IDXS = [
    0.,   0.1875,   0.75,   1.6875,   3.,   4.6875,
    6.75,   9.1875,  12.,  15.1875,  18.75,  22.6875,
    27.,  31.6875,  36.75,  42.1875,  48.,  54.1875,
    60.75,  67.6875,  75.,  82.6875,  90.75,  99.1875,
    108., 117.1875, 126.75, 136.6875, 147., 157.6875,
    168.75, 180.1875, 192.]


PLAN_MEAN = 0
PLAN_STD = 1
PLAN_X = 0
PLAN_Y = 1
LANE_MEAN = 0
LANE_Y = 0


def extract_preds(res):
    plan_start_idx = 0
    plan_end_idx = 4955

    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528

    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

    road_start_idx = lane_lines_prob_end_idx
    road_end_idx = road_start_idx + 264

    plan = res[:, plan_start_idx:plan_end_idx]
    paths = np.array(np.split(plan, 5, axis=1))

    paths = paths.squeeze()  # (5, 991)
    path_probs = paths[:, -1]  # (5, 1)
    paths = paths[:, :-1].reshape(5, 2, 33, 15)  # (5, 2, 33, 15)

    best_idx = np.argmax(path_probs, axis=0)
    best_path = paths[best_idx]

    lanes = res[:, lanes_start_idx:lanes_end_idx]
    lane_road = res[:, road_start_idx:road_end_idx]

    lanes_flat = lanes.flatten()
    ll_t = lanes_flat[0:66]
    ll_t2 = lanes_flat[66:132]
    outer_left_lane, outer_left_lane_std = seperate_points_and_std_values(ll_t)
    inner_left_lane, inner_left_lane_std = seperate_points_and_std_values(ll_t2)

    l_t = lanes_flat[132:198]
    l_t2 = lanes_flat[198:264]
    outer_right_lane, outer_right_lane_std = seperate_points_and_std_values(l_t2)
    inner_right_lane, inner_right_lane_std = seperate_points_and_std_values(l_t)

    road_flat = lane_road.flatten()
    roadr_t = road_flat[0:66]
    roadr_t2 = road_flat[66:132]
    left_road_edge, left_road_edge_std = seperate_points_and_std_values(roadr_t)
    right_road_edge, right_road_edge_std = seperate_points_and_std_values(roadr_t2)

    # r_t = lanes_flat[264:330]
    # r_t2 = lanes_flat[330:396]
    # points_r_t, std_r_t = seperate_points_and_std_values(r_t)
    # points_r_t2, std_r_t2 = seperate_points_and_std_values(r_t2)

    # rr_t = lanes_flat[396:462]
    # rr_t2 = lanes_flat[462:528]
    # points_rr_t, std_rr_t = seperate_points_and_std_values(rr_t)
    # points_rr_t2, std_rr_t2 = seperate_points_and_std_values(rr_t2)

    # roadl_t = df_road[132:198]
    # roadl_t2 = df_road[198:264]
    # points_roadl_t, std_rl_t = seperate_points_and_std_values(roadl_t)
    # points_roadl_t2, std_rl_t2 = seperate_points_and_std_values(roadl_t2)

    # lanelines with std
    lanelines = [
        (left_road_edge, left_road_edge_std, 'yellow'),
        # (outer_left_lane,  outer_left_lane_std, 'sandybrown'),
        (inner_left_lane,  inner_left_lane_std, 'white'),
        (inner_right_lane, inner_right_lane_std, 'white'),
        # (outer_right_lane, outer_right_lane_std, 'sandybrown'),
        (right_road_edge, right_road_edge_std, 'yellow'),

        # (points_roadl_t2, std_rl_t2, 'yellow'),  # useless
        # (points_roadl_t, std_rl_t, 'orange'),  # useless
        # (points_r_t, std_r_t, 'brown'),  # useless
        # (points_r_t2, std_r_t2, 'cyan'),  # useless
        # (points_rr_t, std_rr_t, 'magenta'),  # useless
        # (points_rr_t2, std_rr_t2, 'lime'),  # useless
    ]

    return lanelines, best_path


if __name__ == '__main__':

    path_to_onnx_model = 'train/supercombo.onnx'
    model = onnx.load(path_to_onnx_model)

    input_names = [node.name for node in model.graph.input]
    output_names = [node.name for node in model.graph.output]

    printf('Inputs: ', input_names)
    printf('Outputs: ', output_names)

    # onnxruntime
    providers = ["CPUExecutionProvider"]
    onnxruntime_model = rt.InferenceSession(path_to_onnx_model, providers=providers)

    # pytorch
    device = torch.device('cuda')
    pytorch_model = ConvertModel(model, experimental=False, debug=False).to(device)
    pytorch_model.train(False)  # NOTE: important

    # pytorch_model.Constant_1047.constant= torch.tensor((2,2,66))
    # pytorch_model.Reshape_1048.shape = (2,2,66)
    # pytorch_model.Constant_1049.constant = torch.tensor((2,2,66))
    # pytorch_model.Reshape_1050.shape = (2,2,66)
    # pytorch_model.Constant_1051.constant = torch.tensor((2,2,66))
    # pytorch_model.Reshape_1052.shape = (2,2,66)
    # pytorch_model.Constant_1053.constant = torch.tensor((2,2,66))
    # pytorch_model.Reshape_1054.shape = (2,2,66)
    # pytorch_model.Constant_1057.constant = torch.tensor((2,2,66))
    # pytorch_model.Reshape_1058.shape = (2,2,66)
    # pytorch_model.Constant_1059.constant = torch.tensor((2,2,66))
    # pytorch_model.Reshape_1060.shape = (2,2,66)

    # keras
    keras_model = onnx_to_keras(model, input_names, verbose=False)


    # inference

    comma_recordings_basedir = '/home/nikita/data'
    outs_folder = 'outs'

    train_split = 0.8
    seq_len = 200
    single_frame_batches = False
    batch_size = num_workers = 1
    prefetch_factor = 1

    os.makedirs(outs_folder, exist_ok=True)

    train_dataset = CommaDataset(comma_recordings_basedir, train_split=train_split, seq_len=seq_len,
                                 shuffle=True, single_frame_batches=single_frame_batches, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers, shuffle=False,
                              prefetch_factor=prefetch_factor, persistent_workers=True, collate_fn=None)
    train_loader = BatchDataLoader(train_loader, batch_size=batch_size)
    train_loader = BackgroundGenerator(train_loader)

    recurrent_state_torch = torch.zeros((1, 512), dtype=torch.float32).to(device)
    recurrent_state_onnx = np.zeros((1, 512), dtype=np.float32)
    recurrent_state_keras = np.zeros((1, 512), dtype=np.float32)

    for i, batch in enumerate(train_loader):
        stacked_frames, plans, plans_probs, segment_finished, sequence_finished, bgr_frames = batch

        printf(f'Frames: {stacked_frames.shape}. Plans: {plans.shape}. Plan probs: {plans_probs.shape}. Segment finished: {segment_finished.shape}. Sequence finished: {sequence_finished.shape}')

        bgr_frames = bgr_frames.numpy()
        print('bgr frames:', bgr_frames.shape)

        for t_idx in tqdm(range(seq_len), total=seq_len):

            stacked_frame = stacked_frames[:, t_idx, :, :, :]  # select (1, 12, 128, 256)

            torch_inputs = {
                'input_imgs': stacked_frame.float().to(device),
                'desire': torch.zeros((1, 8), dtype=torch.float32).to(device),
                'traffic_convention': torch.tensor([0, 1], dtype=torch.float32).reshape(1, 2).to(device),
                'initial_state': recurrent_state_torch,
            }

            onnx_inputs = {
                'input_imgs': stacked_frame.numpy().astype(np.float32),
                'desire': np.zeros((1, 8), dtype=np.float32),
                'traffic_convention': np.array([0, 1], dtype=np.float32).reshape(1, 2),
                'initial_state': recurrent_state_onnx,
            }

            keras_inputs = {
                'input_imgs': stacked_frame.numpy().astype(np.float32),
                'desire': np.zeros((1, 8), dtype=np.float32),
                'traffic_convention': np.array([0, 1], dtype=np.float32).reshape(1, 2),
                'initial_state': recurrent_state_keras,
            }

            outs_torch = pytorch_model(**torch_inputs).detach()
            outs_onnx = onnxruntime_model.run(output_names, onnx_inputs)[0]
            outs_keras = keras_model(keras_inputs)

            recurrent_state_torch = outs_torch[:, POSE:]
            recurrent_state_onnx = outs_onnx[:, POSE:]
            recurrent_state_keras = outs_keras[:, POSE:]

            outs_torch = outs_torch.cpu().numpy()
            outs_keras = outs_keras.numpy()
            cv2.imwrite(f'outs/{t_idx}-camera.png', bgr_frames[0, t_idx, :, :, :])

            onnx_lanelines, onnx_path = extract_preds(outs_onnx)
            _, torch_path = extract_preds(outs_torch)
            _, keras_path = extract_preds(outs_keras)

            diff_torch_onnx = np.max(np.abs(torch_path - onnx_path))
            diff_torch_keras = np.max(np.abs(torch_path - keras_path))
            diff_keras_onnx = np.max(np.abs(keras_path - onnx_path))
            max_diff = max(diff_torch_onnx, diff_torch_keras, diff_keras_onnx)

            if max_diff > 0.5:
                print(f'[{t_idx}] largest path diff: {max_diff:.2e}')

            bg_color = '#3c3734'
            fig, ax = plt.subplots(1, 1, figsize=(10, 50))
            fig.patch.set_facecolor(bg_color)
            fig.suptitle(f'Topdown path prediction @ T={t_idx}', fontsize=20, color='white')

            # set tight layout
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            ax.set_facecolor(bg_color)
            ax.set_title(f'Diffs:\nTorch-ONNX: {diff_torch_onnx:.2e}   Torch-Keras: {diff_torch_keras:.2e}   Keras-ONNX: {diff_keras_onnx:.2e}', fontsize=14, color='white')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-1, 150)

            for laneline, conf, color in onnx_lanelines:
                ax.plot(laneline, X_IDXS, linewidth=1, color=color)
                # ax.fill_betweenx(X_IDXS, laneline-conf, laneline+conf, color=color, alpha=0.2)

            plan_sources = ['PyTorch', 'ONNX', 'Keras']
            best_plans = [torch_path, onnx_path, keras_path]

            for model_name, plan in zip(plan_sources, best_plans):

                ax.plot(plan[PLAN_MEAN, :, PLAN_Y], plan[PLAN_MEAN, :, PLAN_X], linewidth=4, alpha=0.5, label=f'Plan by {model_name.capitalize()}')
                ax.fill_betweenx(plan[PLAN_MEAN, :, PLAN_X],
                                 plan[PLAN_MEAN, :, PLAN_Y] - plan[PLAN_STD, :, PLAN_Y],
                                 plan[PLAN_MEAN, :, PLAN_Y] + plan[PLAN_STD, :, PLAN_Y],
                                 alpha=0.2)

            plt.legend()

            fig.savefig(f'outs/{t_idx}-topdown.png')
            fig.clear()
            fig.clf()
            plt.close('all')

        break

    printf('DONE')
