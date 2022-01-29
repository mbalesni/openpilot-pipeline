import warnings

warnings.filterwarnings("ignore", category=UserWarning, message='Length of IterableDataset')
warnings.filterwarnings("ignore", category=UserWarning,
                        message='The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors')
warnings.filterwarnings("ignore", category=UserWarning, message='Using experimental implementation that allows \'batch_size > 1\'')

import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as topt
from dataloader import CommaDataset, BatchDataLoader, BackgroundGenerator, load_transformed_video, configure_worker
from torch.utils.data import DataLoader
import wandb
from timing import Timing, MultiTiming, pprint_stats
from utils import Calibration, draw_path, printf, extract_preds, extract_gt, load_h5
import os
from model import load_model
import gc
import sys
import dotenv
import shutil
import math

dotenv.load_dotenv()


def pprint_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):1d}h {int(minutes):1d}min {int(seconds):1d}s"


# visualizing the model predictions
def visualization(lanelines, roadedges, calib_path, im_rgb):
    plot_img_height, plot_img_width = 480, 640

    rpy_calib = [0, 0, 0]
    X_IDXs = [
        0.,   0.1875,   0.75,   1.6875,   3.,   4.6875,
        6.75,   9.1875,  12.,  15.1875,  18.75,  22.6875,
        27.,  31.6875,  36.75,  42.1875,  48.,  54.1875,
        60.75,  67.6875,  75.,  82.6875,  90.75,  99.1875,
        108., 117.1875, 126.75, 136.6875, 147., 157.6875,
        168.75, 180.1875, 192.]

    calibration_pred = Calibration(rpy_calib, plot_img_width=plot_img_width, plot_img_height=plot_img_height)
    laneline_colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
    vis_image = draw_path(lanelines, roadedges, calib_path[0, :, :3], im_rgb, calibration_pred, X_IDXs, laneline_colors)

    return vis_image

def mean_std(array, eps=1e-10):
    mean = array[:, 0, :, :]
    std = array[:, 1, :, :]

    # we think incoming stds are actually logstds, so exponentiate them to make them non-negative
    std = torch.exp(std)
    # add eps to make positive
    std = torch.add(std, eps)

    return mean, std


def path_laplacian_nll_loss(mean_true, mean_pred, sigma, sigma_clamp: float = 1e-3, loss_clamp: float = 1000.):
    err = torch.abs(mean_true - mean_pred)
    sigma_min = torch.clamp(sigma, min=math.log(sigma_clamp))
    sigma_max = torch.max(sigma, torch.log(1e-6 + err/loss_clamp))
    nll = err * torch.exp(-sigma_max) + sigma_min
    return nll


def path_kl_div_loss(mean1, mean2, std1, std2):
    """
    scratch :Laplace or gaussian likelihood 
    model distillation: gaussian or laplace, KL divergence
    """
    d1 = torch.distributions.laplace.Laplace(mean1, std1)
    d2 = torch.distributions.laplace.Laplace(mean2, std2)
    loss = torch.distributions.kl.kl_divergence(d1, d2).sum(dim=2).sum(dim=1).mean(dim=0)
    return loss


# TODO: vectorize for speedup?
def path_plan_loss(plan_pred, plan_gt, plan_prob_gt):

    paths = plan_pred.reshape(-1, 5, 991)
    path1_pred = paths[:, 0, :-1].reshape(-1, 2, 33, 15)
    path2_pred = paths[:, 1, :-1].reshape(-1, 2, 33, 15)
    path3_pred = paths[:, 2, :-1].reshape(-1, 2, 33, 15)
    path4_pred = paths[:, 3, :-1].reshape(-1, 2, 33, 15)
    path5_pred = paths[:, 4, :-1].reshape(-1, 2, 33, 15)
    path_pred_prob = paths[:, :, -1]

    path1_gt = plan_gt[:, 0, :, :, :]
    path2_gt = plan_gt[:, 1, :, :, :]
    path3_gt = plan_gt[:, 2, :, :, :]
    path4_gt = plan_gt[:, 3, :, :, :]
    path5_gt = plan_gt[:, 4, :, :, :]

    mean_pred_path1, std_pred_path1 = mean_std(path1_pred)
    mean_gt_path1, std_gt_path1 = mean_std(path1_gt)

    mean_pred_path2, std_pred_path2 = mean_std(path2_pred)
    mean_gt_path2, std_gt_path2 = mean_std(path2_gt)

    mean_pred_path3, std_pred_path3 = mean_std(path3_pred)
    mean_gt_path3, std_gt_path3 = mean_std(path3_gt)

    mean_pred_path4, std_pred_path4 = mean_std(path4_pred)
    mean_gt_path4, std_gt_path4 = mean_std(path4_gt)

    mean_pred_path5, std_pred_path5 = mean_std(path5_pred)
    mean_gt_path5, std_gt_path5 = mean_std(path5_gt)

    path1_loss = path_laplacian_nll_loss(mean_gt_path1, mean_pred_path1, std_pred_path1)
    path2_loss = path_laplacian_nll_loss(mean_gt_path2, mean_pred_path2, std_pred_path2)
    path3_loss = path_laplacian_nll_loss(mean_gt_path3, mean_pred_path3, std_pred_path3)
    path4_loss = path_laplacian_nll_loss(mean_gt_path4, mean_pred_path4, std_pred_path4)
    path5_loss = path_laplacian_nll_loss(mean_gt_path5, mean_pred_path5, std_pred_path5)


    # TODO: change to cross-entropy
    path_pred_prob_d = torch.distributions.categorical.Categorical(logits=path_pred_prob)
    path_gt_prob_d = torch.distributions.categorical.Categorical(logits=plan_prob_gt)
    path_prob_loss = torch.distributions.kl.kl_divergence(path_pred_prob_d, path_gt_prob_d).mean(dim=0)

    # MHP loss
    path_head_loss = [path1_loss, path2_loss, path3_loss, path4_loss, path5_loss]
    mask = torch.full((1, 5), 1e-6)

    path_head_loss = torch.tensor(path_head_loss)
    idx = torch.argmin(path_head_loss)

    mask[:, idx] = 1

    path_perhead_loss = torch.mul(path_head_loss, mask)
    path_perhead_loss = path_perhead_loss.sum(dim=1)

    plan_loss = path_perhead_loss + path_prob_loss
    return plan_loss


def train(run, model, train_loader, val_loader, optimizer, scheduler, recurr_warmup, epoch, 
          log_frequency_steps, train_segment_for_viz, val_segment_for_viz, batch_size):

    recurr_input = torch.zeros(batch_size, 512, dtype=torch.float32, device=device, requires_grad=True)
    desire = torch.zeros(batch_size, 8, dtype=torch.float32, device=device)
    traffic_convention = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)
    traffic_convention[:, 1] = 1
    model.train()

    train_loss_accum = 0.0
    segments_finished = True

    start_point = time.time()
    timings = dict()
    multitimings = MultiTiming(timings)
    multitimings.start('batch_load')

    for tr_it, batch in enumerate(train_loader):
        batch_load_time = multitimings.end('batch_load')

        should_log_train = (tr_it+1) % log_frequency_steps == 0
        should_run_valid = (tr_it+1) % val_frequency_steps == 0
        
        printf()
        printf(f"> Got new batch: {batch_load_time:.2f}s - training iteration i am in ", tr_it)
        multitimings.start('train_batch')

        should_backprop = (not recurr_warmup) or (recurr_warmup and not segments_finished)

        stacked_frames, gt_plans, gt_plans_probs, segments_finished = batch
        segments_finished = torch.all(segments_finished)

        loss, recurr_input = train_batch(run, model, optimizer, stacked_frames, gt_plans, gt_plans_probs, desire,
                           traffic_convention, recurr_input, device, timings, should_backprop=should_backprop)

        train_batch_time = multitimings.end('train_batch')
        fps = batch_size * seq_len / train_batch_time
        printf(f"> Batch trained: {train_batch_time:.2f}s (FPS={fps:.2f}).")

        if segments_finished:
            # reset the hidden state for new segments
            printf('Resetting hidden state.')
            recurr_input = recurr_input.zero_().detach()

        train_loss_accum += loss

        if should_run_valid:
            with Timing(timings, 'visualize_preds'):
                visualize_predictions(model, device, train_segment_for_viz, val_segment_for_viz)
            with Timing(timings, 'validate'):
                val_loss = validate(model, val_loader, batch_size, device)

            scheduler.step(val_loss.item())

            checkpoint_save_file = 'commaitr' + date_it + str(val_loss) + '_' + str(epoch+1) + ".pth"
            checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_save_file)
            torch.save(model.state_dict(), checkpoint_save_path)
            model.train()

            wandb.log({
                'validation_loss': val_loss,
            }, commit=False)

        if should_log_train:
            timings['forward_pass']['time'] *= seq_len
            timings['path_plan_loss']['time'] *= seq_len
            
            running_loss = train_loss_accum.item() / log_frequency_steps

            printf()
            printf(f'Epoch {epoch+1}/{epochs}. Done {tr_it+1} steps of ~{train_loader_len}. Running loss: {running_loss:.4f}')
            pprint_stats(timings)
            printf()

            wandb.log({
                'epoch': epoch,
                'train_loss': running_loss,
                'lr': scheduler.optimizer.param_groups[0]['lr'],
                **{f'time_{k}': v['time'] / v['count'] for k, v in timings.items()}
            }, commit=True)

            timings = dict()
            train_loss_accum = 0.0

        multitimings.start('batch_load')

    printf()
    printf(f"Epoch {epoch+1} done! Took {pprint_seconds(time.time() - start_point)}")
    printf()


def visualize_predictions(model, device, train_segment_for_viz, val_segment_for_viz):
    segments_for_viz = [train_segment_for_viz, val_segment_for_viz]

    model.eval()
    with torch.no_grad():

        for i in range(len(segments_for_viz)):

            path_to_segment = segments_for_viz[i]
            printf(f"===>Visualizing predictions: {path_to_segment}")

            recurr_input = torch.zeros(1, 512, dtype=torch.float32, device=device, requires_grad=False)
            desire = torch.zeros(1, 8, dtype=torch.float32, device=device)
            traffic_convention = torch.zeros(1, 2, dtype=torch.float32, device=device)
            traffic_convention[:, 1] = 1

            input_frames, rgb_frames = load_transformed_video(path_to_segment)
            input_frames = input_frames.to(device)

            video_array_pred = np.zeros((rgb_frames.shape[0],rgb_frames.shape[1],rgb_frames.shape[2], rgb_frames.shape[3]), dtype=np.uint8)

            for t_idx in range(rgb_frames.shape[0]): 
                inputs =  {"input_imgs":input_frames[t_idx:t_idx+1],
                                "desire": desire,
                                "traffic_convention": traffic_convention,
                                'initial_state': recurr_input
                                }

                outs = model(**inputs)
                recurr_input = outs[:, 5960:] # refeeding the recurrent state
                preds = outs.detach().cpu().numpy() #(1,6472)

                lanelines, road_edges, best_path = extract_preds(preds)[0]
                im_rgb = rgb_frames[t_idx] 
                vis_image = visualization(lanelines,road_edges,best_path, im_rgb)
                video_array_pred[t_idx:t_idx+1,:,:,:] = vis_image

            video_array_gt = np.zeros((rgb_frames.shape[0],rgb_frames.shape[1],rgb_frames.shape[2], rgb_frames.shape[3]), dtype=np.uint8)
            plan_gt_h5, plan_prob_gt_h5, laneline_gt_h5, laneline_prob_gt_h5, road_edg_gt_h5, road_edgstd_gt_h5 = load_h5(path_to_segment)

            for k in range(plan_gt_h5.shape[0]):

                lane_h5, roadedg_h5, path_h5 = extract_gt(plan_gt_h5[k:k+1], plan_prob_gt_h5[k:k+1], laneline_gt_h5[k:k+1], laneline_prob_gt_h5[k:k+1], road_edg_gt_h5[k:k+1], road_edgstd_gt_h5[k:k+1])[0]
                image_rgb_gt = rgb_frames[k]

                image_gt = visualization(lane_h5, roadedg_h5, path_h5, image_rgb_gt)
                video_array_gt[k:k+1,:,:,:] = image_gt


            video_array_pred = video_array_pred.transpose(0,3,1,2)
            video_array_gt = video_array_gt.transpose(0,3,1,2)
                
            if i == 0:
                video_pred_log_title = "train_pred_video"
                video_gt_log_title = "train_gt_video"
            else:
                video_pred_log_title = "validation_pred_video"
                video_gt_log_title = "validation_gt_video"

            wandb.log({video_pred_log_title: wandb.Video(video_array_pred, fps = 20, format= 'mp4')}, commit=False)
            wandb.log({video_gt_log_title: wandb.Video(video_array_gt, fps = 20, format= 'mp4')}, commit=False)

            del video_array_pred
            del video_array_gt

            gc.collect()


def validate(model, data_loader, batch_size, device):

    model.eval()
    # saving memory by not accumulating activations
    with torch.no_grad():

        val_loss = 0.0

        printf(">>>>>validating<<<<<<<")
        val_itr = None
        recurr_input = torch.zeros(batch_size, 512, dtype=torch.float32, device=device, requires_grad=False)
        for val_itr, val_batch in enumerate(data_loader):

            val_stacked_frames, val_plans, val_plans_probs, segments_finished = val_batch
            segments_finished = torch.all(segments_finished)

            batch_loss, recurr_input = validate_batch(model, val_stacked_frames, val_plans, val_plans_probs, recurr_input, device)
            val_loss += batch_loss

            if segments_finished:
                # reset the hidden state for new segments
                printf('Resetting hidden state.')
                recurr_input.zero_()

            if (val_itr+1) % 10 == 0:
                running_loss = val_loss.item() / (val_itr+1)  # average over entire validation set, no reset as in train
                printf(f'[Validation] Done {val_itr+1} steps of ~{val_loader_len}. Running loss: {running_loss:.4f}')

        if val_itr is not None:
            val_avg_loss = val_loss/(val_itr+1)
            printf(f"Validation Loss: {val_avg_loss:.4f}")

        return val_avg_loss


def train_batch(run, model, optimizer, stacked_frames, gt_plans, gt_plans_probs, desire, traffic_convention, recurr_input, device, timings, should_backprop=True):
    batch_size_empirical = stacked_frames.shape[0]
    seq_len = stacked_frames.shape[1]

    with Timing(timings, 'inputs_to_gpu'):
        stacked_frames = stacked_frames.to(device).float()  # -- (batch_size, seq_len, 12, 128, 256)
        gt_plans = gt_plans.to(device)  # -- (batch_size,seq_len,5,2,33,15)
        gt_plans_probs = gt_plans_probs.to(device)  # -- (batch_size,seq_len,5,1)

    optimizer.zero_grad(set_to_none=True)

    batch_loss = 0.0

    for i in range(seq_len):
        inputs_to_pretained_model = {"input_imgs": stacked_frames[:, i, :, :, :],
                                     "desire": desire,
                                     "traffic_convention": traffic_convention,
                                     'initial_state': recurr_input.clone()  # TODO: why are we cloning recurr_input in 3 places (here, line 428 and line 439?
                                     }

        with Timing(timings, 'forward_pass'):
            outputs = model(**inputs_to_pretained_model)  # -- > [32,6472]

        plan_predictions = outputs[:, :4955].clone()  # -- > [32,4955]
        recurr_out = outputs[:, 5960:].clone()  # -- > [32,512] important to refeed state of GRU

        with Timing(timings, 'path_plan_loss'):
            single_step_loss = path_plan_loss(plan_predictions, gt_plans[:, i, :, :, :, :], gt_plans_probs[:, i, :])

        if i == seq_len - 1:
            # final hidden state in sequence, no need to backpropagate it through time
            pass
        else:
            recurr_input = recurr_out.clone()

        batch_loss += single_step_loss

    complete_batch_loss = batch_loss / seq_len / batch_size_empirical  # mean of losses over batches of sequences

    if should_backprop:
        with Timing(timings, 'backward_pass'):
            complete_batch_loss.backward(retain_graph=True)

    with Timing(timings, 'clip_gradients'):
        torch.nn.utils.clip_grad_norm_(model.parameters(), run.config.grad_clip)

    with Timing(timings, 'optimize_step'):
        optimizer.step()

    loss = complete_batch_loss.detach()  # loss for one iteration
    return loss, recurr_out.detach()


def validate_batch(model, val_stacked_frames, val_plans, val_plans_probs, recurr_input, device):
    batch_size = val_stacked_frames.shape[0]
    seq_len = val_stacked_frames.shape[1]

    desire = torch.zeros(batch_size, 8, dtype=torch.float32, device=device)
    traffic_convention = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)
    traffic_convention[:, 1] = 1

    val_input = val_stacked_frames.float().to(device)
    val_label_path = val_plans.to(device)
    val_label_path_prob = val_plans_probs.to(device)

    val_batch_loss = 0.0

    for i in range(seq_len):
        val_inputs_to_pretained_model = {"input_imgs": val_input[:, i, :, :, :],
                                         "desire": desire,
                                         "traffic_convention": traffic_convention,
                                         "initial_state": recurr_input}

        val_outputs = model(**val_inputs_to_pretained_model)  # --> [32,6472]
        recurr_input = val_outputs[:, 5960:].clone()  # --> [32,512] important to refeed state of GRU
        val_path_prediction = val_outputs[:, :4955].clone()  # --> [32,4955]

        single_val_loss = path_plan_loss(
            val_path_prediction, val_label_path[:, i, :, :, :, :], val_label_path_prob[:, i, :])

        val_batch_loss += single_val_loss

    val_batch_loss = val_batch_loss / seq_len / batch_size
    return val_batch_loss.detach(), recurr_input


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("=>Using '{}' for computation.".format(device))


    # NOTE: important for data loader
    torch.multiprocessing.set_start_method('spawn')
    torch.autograd.set_detect_anomaly(False)

    print("=>Initializing CLI args")
    # CLI parser
    parser = argparse.ArgumentParser(description='Args for comma supercombo train pipeline')
    parser.add_argument("--batch_size", type=int, default=28, help="batch size")
    parser.add_argument("--date_it", type=str, required=True, help="run date/name")  # "16Jan_1_seg"
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip norm")
    parser.add_argument("--log_frequency", type=int, default=100, help="log to wandb every this many steps")
    parser.add_argument("--recordings_basedir", type=dir_path, default="/gpfs/space/projects/Bolt/comma_recordings", help="path to base directory with recordings")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--split", type=float, default=0.94, help="train/val split")
    parser.add_argument("--val_frequency", type=int, default=400, help="run validation every this many steps")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument('--recurr_warmup', dest='recurr_warmup', action='store_true')
    parser.add_argument('--no_recurr_warmup', dest='recurr_warmup', action='store_false')
    parser.add_argument("--l2_lambda", type=float, default=1e-4, help="weight decay rate")
    parser.add_argument("--lrs_thresh", type=float, default=1e-4, help="lrs threshold")
    parser.add_argument("--lrs_min", type=float, default=1e-6, help="lrs min")
    parser.add_argument("--lrs_factor", type=float, default=0.75, help="lrs factor")
    parser.add_argument("--lrs_patience", type=int, default=3, help="lrs patience")
    parser.add_argument("--seq_len", type=int, default=100, help="sequence length")
    parser.add_argument("--no_wandb", dest="no_wandb", action="store_true", help="disable wandb")
    parser.set_defaults(recurr_warmup=True)
    args = parser.parse_args()
 
    # for reproducibility
    torch.manual_seed(args.seed)

    date_it = args.date_it
    train_run_name = date_it
    comma_recordings_basedir = args.recordings_basedir
    path_to_supercombo = '../common/models/supercombo.onnx'

    checkpoints_dir = './nets/checkpoints'
    result_model_dir = './nets/model_itr'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(result_model_dir, exist_ok=True)

    # Hyperparams
    batch_size = num_workers = args.batch_size  # MUST BE batch_size == num_workers
    assert batch_size == num_workers, 'Batch size must be equal to number of workers'

    epochs = args.epochs
    l2_lambda = args.l2_lambda
    log_frequency_steps = args.log_frequency
    lr = args.lr
    lrs_cd = 0
    lrs_factor = args.lrs_factor
    lrs_min = args.lrs_min
    lrs_patience = args.lrs_patience
    lrs_thresh = args.lrs_thresh
    prefetch_factor = 2
    recurr_warmup = args.recurr_warmup
    seq_len = args.seq_len
    train_val_split = args.split
    val_frequency_steps = args.val_frequency

    # only this part of the netwrok is currently trained.
    pathplan_layer_names = ["Gemm_959", "Gemm_981", "Gemm_983", "Gemm_1036"]

    # wandb init
    run = wandb.init(entity=os.environ['WANDB_ENTITY'], project=os.environ['WANDB_PROJECT'], name=train_run_name, mode='offline' if args.no_wandb else 'online')

    # Load data and split in test and train
    printf("=>Loading data")
    printf("=>Preparing the dataloader")
    printf(f"=>Batch size is {batch_size}")

    train_dataset = CommaDataset(comma_recordings_basedir, batch_size=batch_size, train_split=train_val_split, seq_len=seq_len,
                                 shuffle=True, seed=42)
    train_segment_for_viz = os.path.dirname(train_dataset.hevc_file_paths[train_dataset.segment_indices[0]]) # '/home/nikita/data/2021-09-14--09-19-21/2'
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers, shuffle=False, prefetch_factor=prefetch_factor,
                              persistent_workers=True, collate_fn=None, worker_init_fn=configure_worker)
    train_loader = BatchDataLoader(train_loader, batch_size=batch_size)
    train_loader_len = len(train_loader)
    train_loader = BackgroundGenerator(train_loader)

    val_dataset = CommaDataset(comma_recordings_basedir, batch_size=batch_size, train_split=train_val_split, seq_len=seq_len,
                               validation=True, shuffle=True, seed=42)
    val_segment_for_viz = os.path.dirname(val_dataset.hevc_file_paths[val_dataset.segment_indices[0]]) # '/home/nikita/data/2021-09-19--10-22-59/18'
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=num_workers, shuffle=False, prefetch_factor=prefetch_factor,
                            persistent_workers=True, collate_fn=None, worker_init_fn=configure_worker)
    val_loader = BatchDataLoader(val_loader, batch_size=batch_size)
    val_loader_len = len(val_loader)
    val_loader = BackgroundGenerator(val_loader)

    printf('Train visualization segment:', train_segment_for_viz)
    printf('Validation visualization segment:', val_segment_for_viz)

    os.makedirs('tmp', exist_ok=True)
    # shutil.copytree(train_segment_for_viz, 'tmp/train_segment_for_viz')
    # shutil.copytree(val_segment_for_viz, 'tmp/val_segment_for_viz')
    train_segment_for_viz = 'tmp/train_segment_for_viz'
    val_segment_for_viz = 'tmp/val_segment_for_viz'


    printf('Batches in train_loader:', train_loader_len)
    printf('Batches in val_loader:', val_loader_len)

    printf("=>Loading the model")
    comma_model = load_model(path_to_supercombo, trainable_layers=pathplan_layer_names)
    comma_model = comma_model.to(device)

    wandb.watch(comma_model) # Log the gradients  

    param_group = comma_model.parameters()
    optimizer = topt.Adam(param_group, lr, weight_decay=l2_lambda)
    scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lrs_factor, patience=lrs_patience,
                                                    threshold=lrs_thresh, verbose=True, min_lr=lrs_min,
                                                    cooldown=lrs_cd)

    with run:
        printf("=>Run parameters: \n")
        for arg in vars(args):
            wandb.config.update({arg: getattr(args, arg)})
            printf(arg, getattr(args, arg))
        printf()

        printf("=====>Starting to train")
        with torch.autograd.profiler.profile(enabled=False):
            with torch.autograd.profiler.emit_nvtx(enabled=False, record_shapes=False):
                for epoch in tqdm(range(epochs)):
                    train(run, comma_model, train_loader, val_loader, optimizer, scheduler,
                        recurr_warmup, epoch, log_frequency_steps,
                        train_segment_for_viz, val_segment_for_viz, batch_size)

        result_model_save_path = os.path.join(result_model_dir, train_run_name + '.pth')
        torch.save(comma_model.state_dict(), result_model_save_path)
        printf("Saved trained model")
        printf("training_finished")

    sys.exit(0)
