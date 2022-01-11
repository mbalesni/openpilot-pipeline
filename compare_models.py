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
from matplotlib.cm import get_cmap


def printf(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)


def color_thresh(x, threshold=1e-3):
    c = colorama.Fore.RED if x >= threshold else colorama.Fore.WHITE
    return f'{c}{x}'


def seperate_points_and_std_values(path):
    points_indices = np.arange(0, path.shape[-1], 2)
    std_indices = np.arange(1, path.shape[-1], 2)

    points = path[:, points_indices]
    std_values = path[:, std_indices]

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
    # N is batch_size

    plan_start_idx = 0
    plan_end_idx = 4955

    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528

    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

    road_start_idx = lane_lines_prob_end_idx
    road_end_idx = road_start_idx + 264

    plan = res[:, plan_start_idx:plan_end_idx]  # (N, 4955)
    paths = np.array(np.split(plan, 5, axis=1)).reshape(-1, 5, 991)  # (N, 5, 991)
    path_probs = paths[:, :, -1]  # (N, 5)
    paths = paths[:, :, :-1].reshape(-1, 5, 2, 33, 15)  # (N, 5, 2, 33, 15)

    best_idx = np.argmax(path_probs, axis=1)[0]  # (N,)
    best_path = paths[:, best_idx, ...]  # (N, 2, 33, 15)

    lanes = res[:, lanes_start_idx:lanes_end_idx]
    lane_road = res[:, road_start_idx:road_end_idx]

    ll_t = lanes[:, 0:66]

    ll_t2 = lanes[:, 66:132]
    outer_left_lane, outer_left_lane_std = seperate_points_and_std_values(ll_t)
    inner_left_lane, inner_left_lane_std = seperate_points_and_std_values(ll_t2)

    l_t = lanes[:, 132:198]
    l_t2 = lanes[:, 198:264]
    outer_right_lane, outer_right_lane_std = seperate_points_and_std_values(l_t2)
    inner_right_lane, inner_right_lane_std = seperate_points_and_std_values(l_t)

    roadr_t = lane_road[:, 0:66]
    roadr_t2 = lane_road[:, 66:132]
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


def plot_errors_logs(errors_log, batch_sizes, save_path):

    os.makedirs('out', exist_ok=True)

    sample_batch_size = batch_sizes[0]
    op_types = list(errors_log[sample_batch_size][0].keys())
    op_types.sort()

    plots = {}  # dict of (num_op_types, seq_len, 2) arrays, indexed by batch_size

    for batch_size in batch_sizes:
        plots[batch_size] = []

        for op_type in op_types:
            plots[batch_size].append([])

            for frame_idx in range(len(errors_log[batch_size])):

                op_sum_err = np.sum(errors_log[batch_size][frame_idx][op_type])
                op_avg_err = np.mean(errors_log[batch_size][frame_idx][op_type])
                op_stacked_err = np.array([op_sum_err, op_avg_err])
                plots[batch_size][-1].append(op_stacked_err)

            plots[batch_size][-1] = np.array(plots[batch_size][-1])

        plots[batch_size] = np.array(plots[batch_size])

    are_equal = np.allclose(plots[batch_sizes[0]], plots[batch_sizes[1]], rtol=1e-3, atol=1e-4)
    assert are_equal, 'errors for diff batch sizes are not equal'
    print('max diff:', np.max(np.abs(plots[batch_sizes[0]] - plots[batch_sizes[1]])))

    colormap_name = "tab20"
    cmap = get_cmap(colormap_name)  
    colors = cmap.colors

    x = np.arange(seq_len)

    fig, axes = plt.subplots(2, 2, figsize=(30,10))
    fig.suptitle('Errors by NN layer type', fontsize=22)

    reductions = ['Total', 'Average (per instance)']

    for i in range(2):
        for j, (batch_size, bs_errors) in enumerate(plots.items()):
            
            axes[i, j].set_prop_cycle(color=colors)

            if i == 0:
                axes[i, j].set_title(f'Batch size: {batch_size}', fontsize=16)

            if j == 0:
                axes[i, j].set_ylabel(f'{reductions[i]} Error', fontsize=16)

            axes[i, j].set_xlabel('Frame', fontsize=16)
            axes[i, j].set_yscale('log')

            axes[i, j].stackplot(x, bs_errors[..., i], labels=op_types)
            axes[i, j].legend()

    fig.tight_layout()
    fig.savefig(save_path)


def plot_model_preds(data_obj, seq_len, batch_sizes):
    os.makedirs('out', exist_ok=True)

    for t_idx in tqdm(range(seq_len), total=seq_len):
        batch_plots = [list_of_plots[t_idx] for list_of_plots in data_obj]

        bg_color = '#3c3734'
        fig, axes = plt.subplots(1, len(batch_sizes), figsize=(len(data_obj)*10, 50))
        fig.patch.set_facecolor(bg_color)
        fig.suptitle(f'Path prediction diff @ T={t_idx}', fontsize=20, color='white')
        fig.tight_layout()

        for ind, ax in enumerate(axes):
            batch_size = batch_sizes[ind]
            plot_data = batch_plots[ind]

            diff = plot_data['diff']
            torch_lanelines = plot_data['torch_lanelines']
            keras_lanelines = plot_data['keras_lanelines']
            plan_sources = plot_data['plan_sources']
            best_plans = plot_data['best_plans']

            ax.set_facecolor(bg_color)
            ax.set_title(f'Diff (batch_size={batch_size}): {diff:.1e}', fontsize=14, color='white')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-1, 150)

            for laneline, conf, color in torch_lanelines:
                ax.plot(laneline[sample_idx], X_IDXS, linewidth=1, color='white')

            for laneline, conf, color in keras_lanelines:
                ax.plot(laneline[sample_idx], X_IDXS, linewidth=1, color='limegreen')

            for model_name, plan in zip(plan_sources, best_plans):

                label = f'Plan by {model_name.capitalize()}'
                if model_name != 'torch':
                    label += f' ({diff:.1e})'

                ax.plot(plan[PLAN_MEAN, :, PLAN_Y], plan[PLAN_MEAN, :, PLAN_X], linewidth=4, alpha=0.5, label=label)
                ax.fill_betweenx(plan[PLAN_MEAN, :, PLAN_X],
                                plan[PLAN_MEAN, :, PLAN_Y] - plan[PLAN_STD, :, PLAN_Y],
                                plan[PLAN_MEAN, :, PLAN_Y] + plan[PLAN_STD, :, PLAN_Y],
                                alpha=0.2)

        plt.legend()

        fig.savefig(f'outs/{t_idx}-model_preds.png')
        fig.clear()
        fig.clf()
        plt.close('all')


if __name__ == '__main__':

    comma_recordings_basedir = '/home/nikita/data'
    path_to_onnx_model = 'train/supercombo.onnx'
    outs_folder = 'outs'

    train_split = 0.8
    seq_len = 10
    single_frame_batches = False
    prefetch_factor = 1
    debug = True

    os.makedirs(outs_folder, exist_ok=True)
    model = onnx.load(path_to_onnx_model)

    node_nums_to_node = {node.output[0]: node for node in model.graph.node}

    input_names = [node.name for node in model.graph.input]
    output_names = [node.name for node in model.graph.output]

    printf('Inputs: ', input_names)
    printf('Outputs: ', output_names)

    errors = []
    model_logs = []

    # batch_sizes = [1, 8]
    # batch_sizes = [1, 4]
    batch_sizes = [1, 2]

    errors_logs = {}


    for batch_size in batch_sizes:
        num_workers = batch_size

        print('-- TEST BATCH SIZE:', batch_size)

        errors.append([])
        model_logs.append([])
        errors_logs[batch_size] = []


        # onnxruntime
        # providers = ["CPUExecutionProvider"]
        # onnxruntime_model = rt.InferenceSession(path_to_onnx_model, providers=providers)

        # pytorch
        device = torch.device('cuda:1')
        pytorch_model = ConvertModel(model, experimental=True, debug=debug).to(device)
        pytorch_model.requires_grad_(False)
        pytorch_model.train(False)  # NOTE: important

        pytorch_model.Constant_1047.constant= torch.tensor((batch_size, 2, 66))
        pytorch_model.Reshape_1048.shape = (batch_size, 2, 66)
        pytorch_model.Constant_1049.constant = torch.tensor((batch_size, 2, 66))
        pytorch_model.Reshape_1050.shape = (batch_size, 2, 66)
        pytorch_model.Constant_1051.constant = torch.tensor((batch_size, 2, 66))
        pytorch_model.Reshape_1052.shape = (batch_size, 2, 66)
        pytorch_model.Constant_1053.constant = torch.tensor((batch_size, 2, 66))
        pytorch_model.Reshape_1054.shape = (batch_size, 2, 66)
        pytorch_model.Constant_1057.constant = torch.tensor((batch_size, 2, 66))
        pytorch_model.Reshape_1058.shape = (batch_size, 2, 66)
        pytorch_model.Constant_1059.constant = torch.tensor((batch_size, 2, 66))
        pytorch_model.Reshape_1060.shape = (batch_size, 2, 66)

        # keras
        keras_model = onnx_to_keras(model, input_names, verbose=False)


        # inference
        train_dataset = CommaDataset(comma_recordings_basedir, train_split=train_split, seq_len=seq_len,
                                    shuffle=True, single_frame_batches=single_frame_batches, seed=42)
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers, shuffle=False,
                                prefetch_factor=prefetch_factor, persistent_workers=True, collate_fn=None)
        train_loader = BatchDataLoader(train_loader, batch_size=batch_size)
        train_loader = BackgroundGenerator(train_loader)

        recurrent_state_torch = torch.zeros((batch_size, 512), dtype=torch.float32).to(device)
        recurrent_state_onnx = np.zeros((1, 512), dtype=np.float32)
        recurrent_state_keras = np.zeros((1, 512), dtype=np.float32)

        for i, batch in enumerate(train_loader):
            stacked_frames, plans, plans_probs, segment_finished, sequence_finished, bgr_frames = batch

            printf(f'Frames: {stacked_frames.shape}. Plans: {plans.shape}. Plan probs: {plans_probs.shape}. Segment finished: {segment_finished.shape}. Sequence finished: {sequence_finished.shape}')

            bgr_frames = bgr_frames.numpy()  # (batch_size, seq_len, 874, 1164, 3)

            for t_idx in tqdm(range(seq_len), total=seq_len):

                stacked_frame = stacked_frames[:, t_idx, :, :, :]  # (batch_size, 12, 128, 256) selects the `t_idx`-th frame for each of `batch_size` segments 
                sample_idx = 0  # keep 0, used to select a single sample for keras/onnx

                torch_inputs = {
                    'input_imgs': stacked_frame.float().to(device),
                    'desire': torch.zeros((batch_size, 8), dtype=torch.float32).to(device),
                    'traffic_convention': torch.tensor([[0, 1]] * batch_size, dtype=torch.float32).to(device),
                    'initial_state': recurrent_state_torch,
                }

                onnx_inputs = {
                    'input_imgs': np.expand_dims(stacked_frame.numpy()[sample_idx, ...], 0).astype(np.float32),
                    'desire': np.zeros((1, 8), dtype=np.float32),
                    'traffic_convention': np.array([0, 1], dtype=np.float32).reshape(1, 2),
                    'initial_state': recurrent_state_onnx,
                }

                keras_inputs = {
                    'input_imgs': np.expand_dims(stacked_frame.numpy()[sample_idx, ...], 0).astype(np.float32),
                    'desire': np.zeros((1, 8), dtype=np.float32),
                    'traffic_convention': np.array([0, 1], dtype=np.float32).reshape(1, 2),
                    'initial_state': recurrent_state_keras,
                }


                # TODO: so path predictions have higher errors for batch_size > 0 (e.g. see `outs/127-model_preds.png`),
                # but the the plots for errors by layer type (e.g. see `errors_by_op.png`) say the errors are the same for batch_size=1 and batch_size=4.
                # Need to diagnose this.
                # 1. Are errors really higher for batch_size > 1?
                # 2. If so, which layers contribute to this?
                # 3. Any ideas for fixes?


                if debug:
                    desired_output_layers = list(set(node_nums_to_node.keys()) - set(input_names))

                    keras_layer_outs = get_activations(keras_model, list(keras_inputs.values()), layer_names=None, output_format='simple', auto_compile=True)
                    keras_layer_outs = {k: v for k, v in keras_layer_outs.items() if k in desired_output_layers}  # indexed by layer num

                    torch_layer_outs, diffs = pytorch_model(**torch_inputs, debug_activations=keras_layer_outs)  # indexed by layer num
                    # outs_onnx = onnxruntime_model.run(output_names, onnx_inputs)[0]

                    recurrent_state_torch = torch_layer_outs[:, POSE:]
                    recurrent_state_keras = keras_layer_outs['outputs'][:, POSE:]
                    # recurrent_state_onnx = outs_onnx[:, POSE:]

                    errors_logs[batch_size].append({})

                    for node_num, diff in diffs.items():

                        nodename = node_nums_to_node[node_num].name

                        # save diff
                        node_op = node_nums_to_node[node_num].op_type
                        if node_op not in errors_logs[batch_size][-1]:
                            errors_logs[batch_size][-1][node_op] = []
                        errors_logs[batch_size][-1][node_op].append(diff)

                        # TODO: plot diff for each node in time

                    torch_outs = torch_layer_outs.detach().cpu().numpy()
                    keras_outs = keras_layer_outs['outputs']
                else:
                    torch_outs = pytorch_model(**torch_inputs)
                    keras_outs = keras_model(keras_inputs)

                    recurrent_state_torch = torch_outs[:, POSE:]
                    recurrent_state_keras = keras_outs[:, POSE:]

                    torch_outs = torch_outs.detach().cpu().numpy()
                    keras_outs = keras_outs.numpy()


                cv2.imwrite(f'outs/{t_idx}-camera.png', bgr_frames[sample_idx, t_idx, :, :, :])

                torch_lanelines, torch_path = extract_preds(torch_outs)
                keras_lanelines, keras_path = extract_preds(keras_outs)
                diff_torch_keras = np.max(np.abs(torch_path[sample_idx] - keras_path[sample_idx]))
                plan_sources = ['PyTorch', 'Keras']
                best_plans = [torch_path[sample_idx], keras_path[sample_idx]]

                model_preds = {
                    'diff': diff_torch_keras,
                    'torch_lanelines': torch_lanelines,
                    'keras_lanelines': keras_lanelines,
                    'plan_sources': plan_sources,
                    'best_plans': best_plans,
                }

                errors[-1].append(diff_torch_keras)
                model_logs[-1].append(model_preds)

            break



    if debug:
        plot_errors_logs(errors_logs, batch_sizes, 'errors_by_op.png')

    # plot_model_preds(model_logs, seq_len, batch_sizes)

    printf('DONE')
