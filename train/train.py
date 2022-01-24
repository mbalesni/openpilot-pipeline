import argparse
import itertools
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as topt
from dataloader import CommaDataset, BatchDataLoader, BackgroundGenerator, load_transformed_video, configure_worker
from torch.utils.data import DataLoader
import wandb
from timing import Timing, pprint_stats
from utils import Calibration, draw_path, printf, extract_preds, extract_gt, load_h5
import os
import warnings
from model import load_model

# PyTorch assumes each DataLoader worker to return a batch, but we return a single sample, so the length warning is a false alarm.
warnings.filterwarnings("ignore", category=UserWarning, message='Length of IterableDataset')
warnings.filterwarnings("ignore", category=UserWarning, message='The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors')
warnings.filterwarnings("ignore", category=UserWarning, message='Using experimental implementation that allows \'batch_size > 1\'')


torch.autograd.set_detect_anomaly(True)

# logger class for Wandb
class Logger:
    def __init__(self, prefix):
        self.cur_ep = 0
        self.prefix = prefix

    def plotTr(self, loss, lr, time, epoch=-1):
        # TODO: @nikebless look into this logic, looks suspicious.
        if epoch == -1:
            self.cur_ep += 1
        else:
            self.cur_ep = epoch
        wandb.log({"{}_Loss".format(self.prefix): loss,
                   "{}_Time".format(self.prefix): time,
                   "{}_lr".format(self.prefix): lr},
                  step=self.cur_ep)


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

#     print(vis_image.shape)
    return vis_image

# Loss functions:


def mean_std(array):

    mean = array[:, 0, :, :]
    # mean = mean

    std = array[:, 1, :, :]
    std = torch.exp(std)  # to escape the negative values

    """
    check if the resulting mean and std have values (0,greate than zero]
    if the resulting values are too small(equivalent to zero) add epsilon to fullfil the bound conditions for pytorch distribution
    """
    eps = 1e-10
    if torch.any(torch.absolute(mean) < eps):
        mean = torch.add(mean, eps)
    elif torch.any(torch.absolute(std) < eps):
        std = torch.add(std, eps)

    return mean, std


def calculate_path_loss(mean1, mean2, std1, std2):
    """
    scratch :Laplace or gaussian likelihood 
    model distillation: gaussian or laplace, KL divergence
    """
    d1 = torch.distributions.laplace.Laplace(mean1, std1)
    d2 = torch.distributions.laplace.Laplace(mean2, std2)
    loss = torch.distributions.kl.kl_divergence(d1, d2).sum(dim=2).sum(dim=1).mean(dim=0)
    return loss


def path_plan_loss(plan_pred, plan_gt, plan_prob_gt, batch_size, mhp_loss=False):

    paths = plan_pred.reshape(batch_size, 5, 991)
    path1_pred = paths[:, 0, :-1].reshape(batch_size, 2, 33, 15)
    path2_pred = paths[:, 1, :-1].reshape(batch_size, 2, 33, 15)
    path3_pred = paths[:, 2, :-1].reshape(batch_size, 2, 33, 15)
    path4_pred = paths[:, 3, :-1].reshape(batch_size, 2, 33, 15)
    path5_pred = paths[:, 4, :-1].reshape(batch_size, 2, 33, 15)
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

    path1_loss = calculate_path_loss(mean_pred_path1, mean_gt_path1, std_pred_path1, std_gt_path1)
    path2_loss = calculate_path_loss(mean_pred_path2, mean_gt_path2, std_pred_path2, std_gt_path2)
    path3_loss = calculate_path_loss(mean_pred_path3, mean_gt_path3, std_pred_path3, std_gt_path3)
    path4_loss = calculate_path_loss(mean_pred_path4, mean_gt_path4, std_pred_path4, std_gt_path4)
    path5_loss = calculate_path_loss(mean_pred_path5, mean_gt_path5, std_pred_path5, std_gt_path5)

    path_pred_prob_d = torch.distributions.categorical.Categorical(logits=path_pred_prob)
    path_gt_prob_d = torch.distributions.categorical.Categorical(logits=plan_prob_gt)
    path_prob_loss = torch.distributions.kl.kl_divergence(path_pred_prob_d, path_gt_prob_d).mean(dim=0)

    # naive loss
    plan_loss = path1_loss + path2_loss + path3_loss + path4_loss + path5_loss + path_prob_loss

    '''
    # winner-take-all loss
    path_head_loss = [path1_loss, path2_loss, path3_loss, path4_loss, path5_loss]
    mask = torch.full((1, 5), 1e-6)

    path_head_loss = torch.tensor(path_head_loss)
    idx = torch.argmin(path_head_loss)

    mask[:, idx] = 1

    path_perhead_loss = torch.mul(path_head_loss, mask)
    path_perhead_loss = path_perhead_loss.sum(dim=1)

    plan_loss = path_perhead_loss + path_prob_loss

    # TODO: confirm to add and find the path_prob_loss via kldiv or crossentropy
    # (@gautam: as tambet suggested us to go with distillation, so in that particular case crossentropy is not valid, we have to go with kldiv. )
    '''
    return plan_loss

if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("=> using '{}' for computation.".format(device))

    # NOTE: important for data loader
    torch.multiprocessing.set_start_method('spawn')

    print("=>intializing CLI args")
    # CLI parser
    parser = argparse.ArgumentParser(description='Args for comma supercombo train pipeline')
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(args.seed)
    print("=>seed={}".format(args.seed))

    printf("=>intializing hyperparams")

    date_it = "16Jan_1_seg"
    train_run_name = "onnx_gen_gt_comma_pipeline_" + date_it
    comma_recordings_basedir = "/gpfs/space/projects/Bolt/comma_recordings"
    path_to_supercombo = '../common/models/supercombo.onnx'

    checkpoints_dir = './nets/checkpoints'
    result_model_dir = './nets/model_itr'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(result_model_dir, exist_ok=True)

    # Hyperparams
    lr = 0.001
    diff_lr = False
    recurr_warmup = True
    l2_lambda = 1e-4
    lrs_factor = 0.75
    lrs_patience = 3
    lrs_cd = 0
    lrs_thresh = 1e-4
    lrs_min = 1e-6

    epochs = 10
    check_val_epoch = 2
    batch_size = num_workers = args.batch_size  # MUST BE batch_size == num_workers
    assert batch_size == num_workers, 'Batch size must be equal to number of workers'
    split_per = 0.98
    prefetch_factor = 2
    seq_len = 100
    prefetch_warmup_time = 10  # seconds wait before starting iterating

    # only this part of the netwrok is currently trained.
    pathplan_layer_names = ["Gemm_959", "Gemm_981", "Gemm_983", "Gemm_1036"]

    tr_logger = Logger("train")
    val_logger = Logger("validation")

    # wandb init
    # run = wandb.init(project="test-project", entity="openpilot_project", train_run_name = train_run_name, reinit= True, tags= ["supercombbo pretrain"])

    # Load data and split in test and train
    printf("=>Loading data")
    printf("=>Preparing the dataloader")
    printf(f"=>Batch size is {batch_size}")

    train_dataset = CommaDataset(comma_recordings_basedir, batch_size=batch_size, train_split=split_per, seq_len=seq_len,
                                 shuffle=True, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers, shuffle=False, prefetch_factor=prefetch_factor,
                              persistent_workers=True, collate_fn=None, worker_init_fn=configure_worker)
    train_loader = BatchDataLoader(train_loader, batch_size=batch_size)
    train_loader_len = len(train_loader)
    train_loader = BackgroundGenerator(train_loader)

    val_dataset = CommaDataset(comma_recordings_basedir, batch_size=batch_size, train_split=split_per, seq_len=seq_len, 
                               validation=True, shuffle=True, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=num_workers, shuffle=False, prefetch_factor=prefetch_factor,
                            persistent_workers=True, collate_fn=None, worker_init_fn=configure_worker)
    val_loader = BatchDataLoader(val_loader, batch_size=batch_size)
    val_loader_len = len(val_loader)
    val_loader = BackgroundGenerator(val_loader)

    print('Batches in train_loader:', train_loader_len)
    print('Batches in val_loader:', val_loader_len)

    printf("=>Loading the model")
    comma_model = load_model(path_to_supercombo, trainable_layers=pathplan_layer_names)
    comma_model = comma_model.to(device)

    # wandb.watch(comma_model) # Log the network weight histograms

    # Define optimizer and scheduler
    # diff. learning rate for different parts of the network.
    param_group = comma_model.parameters()

    optimizer = topt.Adam(param_group, lr, weight_decay=l2_lambda)
    scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lrs_factor, patience=lrs_patience,
                                                    threshold=lrs_thresh, verbose=True, min_lr=lrs_min,
                                                    cooldown=lrs_cd)

    printf("=====> all the params are successfully loaded")

    # train loop
    # initializing recurrent state by zeros
    recurrent_state = torch.zeros(batch_size, 512, dtype=torch.float32)
    recurrent_state = recurrent_state.to(device)

    # desire and traffic convention is also set to zero
    desire = torch.zeros(batch_size, 8, dtype=torch.float32)
    desire = desire.requires_grad_(True).to(device)

    # LHD
    traffic_convention = torch.zeros(batch_size, 2, dtype=torch.float32)
    traffic_convention[:, 1] = 1
    traffic_convention = traffic_convention.requires_grad_(True).to(device)
     
    #  with run:
    #     # wandb.config.l2 = l2_lambda
    #     # wandb.config.lrs = str(scheduler)
    #     # wandb.config.lr = lr
    #     # wandb.config.batch_size = batch_size
   

    recurr_tmp = torch.zeros(batch_size, 512, dtype=torch.float32)

    for epoch in tqdm(range(epochs)):

        start_point = time.time()
        tr_loss = 0.0
        run_loss = 0.0

        comma_model.train()

        recurr_tmp = torch.zeros(batch_size, 512, dtype=torch.float32)

        batch_load_start = time.time()
        processing_time = 0

        timings = dict()

        for tr_it, batch in enumerate(train_loader):
            fps = batch_size * seq_len / processing_time if processing_time > 0 else 0
            printf()
            printf(f"[Last batch processed: {processing_time:.2f}s (FPS={fps:.2f}). Waited new batch: {time.time() - batch_load_start:.2f}s] - training iteration i am in ", tr_it)

            processing_time_start = time.time()

            with Timing(timings, 'recurr_state_clone'):
                recurr_state = recurrent_state.clone().requires_grad_(True)

            stacked_frames, gt_plans, gt_plans_probs, segment_finished, sequence_finished = batch

            batch_size_empirical = stacked_frames.shape[0]

            with Timing(timings, 'inputs_to_gpu'):
                stacked_frames = stacked_frames.to(device).float()  # -- (batch_size, seq_len, 12, 128, 256)
                gt_plans = gt_plans.to(device)  # -- (batch_size,seq_len,5,2,33,15)
                gt_plans_probs = gt_plans_probs.to(device)  # -- (batch_size,seq_len,5,1)

            with Timing(timings, 'zero_grad'):
                optimizer.zero_grad()
            batch_loss = 0

            for i in range(seq_len):
                with Timing(timings, 'recurr_state_clone'):
                    prev_recurr_state = recurr_state.clone()  # TODO: why are we cloning recurr_state in 3 places (here, line 428 and line 439?

                inputs_to_pretained_model = {"input_imgs": stacked_frames[:, i, :, :, :],
                                             "desire": desire,
                                             "traffic_convention": traffic_convention,
                                             'initial_state': prev_recurr_state
                                             }

                with Timing(timings, 'forward_pass'):
                    outputs = comma_model(**inputs_to_pretained_model)  # -- > [32,6472]

                with Timing(timings, 'plan_preds_clone'):
                    plan_predictions = outputs[:, :4955].clone()  # -- > [32,4955]

                with Timing(timings, 'recurr_state_clone'):
                    recurr = outputs[:, 5960:].clone()  # -- > [32,512] important to refeed state of GRU

                # labels_path_prob[:,i,:,:] -- > [32,5,1]
                # labels_path[:,i,:,:,:,:] --> [32,5,2,33,15]

                with Timing(timings, 'path_plan_loss'):
                    single_step_loss = path_plan_loss(plan_predictions, gt_plans[:, i, :, :, :, :], gt_plans_probs[:, i, :], batch_size)

                # printf("testing single batch Loss:",single_step_loss.item())

                if i == seq_len - 1:
                    pass
                else:
                    with Timing(timings, 'recurr_state_clone'):
                        recurr_state = recurr.clone()

                batch_loss += single_step_loss

            complete_batch_loss = batch_loss / seq_len / batch_size_empirical  # mean of losses over batches of sequences

            if not (recurr_warmup and epoch == 0 and tr_it == 0):
                with Timing(timings, 'backward_pass'):
                    complete_batch_loss.backward(retain_graph=False)

            recurr_state = recurr

            with Timing(timings, 'optimize_step'):
                optimizer.step()

            loss_cpu = complete_batch_loss.detach().clone().item()  # loss for one iteration
            tr_loss += loss_cpu
            run_loss += loss_cpu

            if (tr_it+1) % 10 == 0:
                printf(f'{epoch+1}/{epochs}, step [{tr_it+1} of ~{train_loader_len}], loss: {tr_loss/(tr_it+1):.4f}')

                timings['recurr_state_clone']['time'] *= seq_len
                timings['forward_pass']['time'] *= seq_len
                timings['plan_preds_clone']['time'] *= seq_len
                timings['path_plan_loss']['time'] *= seq_len

                pprint_stats(timings)
                timings = dict()
                printf()
                # TODO: for @nikebless, verify that the running loss is computed correctly

                if (tr_it+1) % 100 == 0:
                    # tr_logger.plotTr( run_loss /100, optimizer.param_groups[0]['lr'], time.time() - start_point )  # add get current learning rate adjusted by the scheduler.
                    with Timing(timings, 'scheduler_step'):
                        scheduler.step(run_loss/100)
                    run_loss = 0.0

            # validation loop
            #validate at every 10000 step every epoch
            if (tr_it + 1) % 400 == 0:

                comma_model.eval()
                # saving memory by not accumulating activations
                with torch.no_grad():

                    """
                    visualization
                    """
                    printf("===> visualizing the predictions")

                    val_video_paths = ['/gpfs/space/projects/Bolt/comma_recordings/comma2k19/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/4',
                                        '/gpfs/space/projects/Bolt/comma_recordings/realdata/2021-09-07--08-22-59/14']

                    for i in range(len(val_video_paths)):

                        input_frames, rgb_frames = load_transformed_video(val_video_paths[i])

                        # print(input_frames.shape, rgb_frames.shape)

                        video_array = np.zeros(
                            ((int(np.round(rgb_frames.shape[0]/batch_size)*batch_size), rgb_frames.shape[1], rgb_frames.shape[2], rgb_frames.shape[3])))
                        # print(video_array.shape)

                        for itr in range(int(np.round(input_frames.shape[0]/batch_size))):  # ---eg. for batch_size 32 skipping 6 frames for video

                            start_indx, end_indx = itr * batch_size, (itr + 1) * batch_size

                            itr_input_frames = input_frames[start_indx:end_indx]
                            itr_rgb_frames = rgb_frames[start_indx:end_indx]

                            inputs = {"input_imgs": itr_input_frames.to(device),
                                        "desire": desire,
                                        "traffic_convention": traffic_convention,
                                        'initial_state': recurr_state
                                        }

                            outs = comma_model(**inputs)

                            preds = outs.detach().cpu().numpy()  # (N,6472)

                            batch_vis_img = np.zeros((preds.shape[0], rgb_frames.shape[1], rgb_frames.shape[2], rgb_frames.shape[3]))

                            for j in range(preds.shape[0]):

                                pred_it = preds[j][np.newaxis, :]
                                lanelines, road_edges, best_path = extract_preds(pred_it)[0]

                                im_rgb = itr_rgb_frames[j]

                                image = visualization(lanelines, road_edges, best_path, im_rgb)

                                batch_vis_img[j] = image

                            video_array[start_indx:end_indx, :, :, :] = batch_vis_img
                       
                        ## groundtruth_visualization ##
                        video_array_gt = np.zeros(((int(np.round(rgb_frames.shape[0]/batch_size)*batch_size),rgb_frames.shape[1],rgb_frames.shape[2], rgb_frames.shape[3])))
                        # plan, plan_prob, lanelines, lanelines_prob, road_edg, road_edg_std,file
                        plan_gt_h5, plan_prob_gt_h5, laneline_gt_h5, laneline_prob_gt_h5, road_edg_gt_h5, road_edgstd_gt_h5, h5_file_object = load_h5(val_video_paths[i])

                        for k in range(plan_gt_h5.shape[0]):
                            
                            lane_h5, roadedg_h5, path_h5 = extract_gt(plan_gt_h5[k:k+1], plan_prob_gt_h5[k:k+1], laneline_gt_h5[k:k+1], laneline_prob_gt_h5[k:k+1], road_edg_gt_h5[k:k+1], road_edgstd_gt_h5[k:k+1])[0]
                            image_rgb_gt = rgb_frames[k]

                            image_gt = visualization(lane_h5, roadedg_h5, path_h5, image_rgb_gt)
                            video_array_gt[k:k+1,:,:,:] = image_gt
                        
                        h5_file_object.close()

                        video_array = video_array.transpose(0, 3, 1, 2)
                        video_array_gt = video_array_gt.transpose(0,3,1,2)
                        
                        if i == 0:
                            video_pred_log_title = "val_video_trainset" + str(epoch)
                            video_gt_log_title = "gt_video_trainset" + str(epoch)
                        else:
                            video_pred_log_title = "val_video_valset" + str(epoch)
                            video_gt_log_title = "gt_video_valset" + str(epoch)
                            
                        # wandb.log({video_pred_log_title: wandb.Video(video_array, fps = 20, format= 'mp4')})
                        # wandb.log({video_gt_log_title: wandb.Video(video_array_gt, fps = 20, format= 'mp4')})
                            
                    val_st_pt = time.time()
                    val_loss_cpu = 0.0

                    printf(">>>>>validating<<<<<<<")

                    for val_itr, val_batch in enumerate(val_loader):

                        val_stacked_frames, val_plans, val_plans_probs, val_segment_finished, val_sequence_finished = val_batch

                        val_input = val_stacked_frames.float().to(device)
                        val_label_path = val_plans.to(device)
                        val_label_path_prob = val_plans_probs.to(device)

                        val_batch_loss = torch.zeros(1, dtype=torch.float32, requires_grad=True).to(device)

                        for i in range(seq_len):
                            val_inputs_to_pretained_model = {"input_imgs": val_input[:, i, :, :, :],
                                                                "desire": desire,
                                                                "traffic_convention": traffic_convention,
                                                                "initial_state": recurr_state}

                            val_outputs = comma_model(**val_inputs_to_pretained_model)  # --> [32,6472]
                            val_path_prediction = val_outputs[:, :4955].clone()  # --> [32,4955]

                            # val_labels_path_prob[:,i,:,:] -- > [32,5,1]
                            # val_labels_path[:,i,:,:,:,:] --> [32,5,2,33,15]

                            single_val_loss = path_plan_loss(
                                val_path_prediction, val_label_path[:, i, :, :, :, :], val_label_path_prob[:, i, :], batch_size)

                            val_batch_loss += single_val_loss

                        val_batch_loss = val_batch_loss/batch_size
                        val_loss_cpu += val_batch_loss.detach().clone().cpu().item()

                        if (val_itr+1) % 10 == 0:
                            printf(f'Epoch:{epoch+1} ,step [{val_itr+1} of ~{val_loader_len}], loss: {val_loss_cpu/(val_itr+1):.4f}')

                    printf(f"Epoch: {epoch+1}, Val Loss: {val_loss_cpu/(val_itr+1):.4f}")

                    val_loss_name = str(val_loss_cpu/(val_itr+1))

                    checkpoint_save_file = 'commaitr' + date_it + val_loss_name + '_' + str(epoch+1) + ".pth"
                    checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_save_file)
                    torch.save(comma_model.state_dict(), checkpoint_save_path)

                    # val_logger.plotTr(val_loss_cpu/(val_itr+1), optimizer.param_groups[0]['lr'], time.time() - val_st_pt)

            processing_time = time.time() - processing_time_start
            batch_load_start = time.time()

        printf(f"Epoch: {epoch+1}, Train Loss: {tr_loss/(tr_it+1)}, time_per_epoch: {time.time() - start_point}")
        # tr_logger.plotTr(tr_loss/(tr_it+1), optimizer.param_groups[0]['lr'], time.time() - start_point )

    result_model_save_path = os.path.join(result_model_dir, train_run_name + '.pth')
    torch.save(comma_model.state_dict(), result_model_save_path)
    printf("Saved trained model")
    printf("training_finished")
