import argparse
import itertools
import time
import numpy as np 
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.optim as topt
from dataloader import CommaLoader
from torch.utils.data import DataLoader
from model import *
from onnx2pytorch import ConvertModel
import onnx
import wandb
torch.autograd.set_detect_anomaly(True)

cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))


#for reproducibility 
seed = np.random.randint(2**16)
torch.manual_seed(seed)
print("=>seed={}".format(seed))

print("=>intializing CLI args")
# CLI parser 
parser = argparse.ArgumentParser(description='Args for comma supercombo train pipeline')
parser.add_argument("--datatype", type=str, default = "", choices=["dummy", "gen_gt"], help= "directory in which the dummy data or generated gt lies" )
parser.add_argument("--phase", type =str, default="train", choices = ["train", "test"])
parser.add_argument("--batch_size", type= int, default=1, help = "batch size")
parser.add_argument("--modeltype", type = str, default = "scratch", choices= ["scratch", "onnx"], help = "choose type of model for train")
args = parser.parse_args()

class Logger:
    def __init__( self, prefix ):
        self.cur_ep = 0
        self.prefix = prefix
        
    def plotTr( self, loss, lr, time, epoch=-1 ):
        if epoch == -1:
            self.cur_ep += 1
        else: self.cur_ep = epoch
        wandb.log( {"{}_Loss".format( self.prefix ): loss,
                    "{}_Time".format( self.prefix ): time,
                    "{}_lr".format( self.prefix ): lr},
                    step=self.cur_ep )

## intializing the object of the logger class 
print("=>intialzing wandb Logger class")

tr_logger = Logger("train")
val_logger = Logger("validation")

print("=>intializing hyperparams")
#Hyperparams
date_it  = "_20dec"
name = "onnx_gen_gt_comma_pipeline_" + date_it
path_comma_recordings = "/gpfs/space/projects/Bolt/comma_recordings"
path_npz_dummy = ["inputdata.npz","gtdata.npz"] # dummy data_path
onnx_path = 'supercombo.onnx'
n_workers = 20
lr = (0.001, 2e-4, 1e-3) ## (lr_conv, lr_gru, lr_outhead)
diff_lr = False
recurr_warmup = True
l2_lambda = (1e-4,1e-4,1e-4) 
lrs_factor = 0.75
lrs_patience = 50
lrs_cd = 50
lrs_thresh = 1e-4
lrs_min = 1e-6

epochs = 10
check_val_epoch =2 
batch_size = args.batch_size
split_per = 0.8

#wandb init
run = wandb.init(project="test-project", entity="openpilot_project", name = name, reinit= True, tags= ["supercombbo pretrain"])

### Load data and split in test and train
print("=>Loading data")
print("=>Preparing the dataloader")

if "dummy" in name:
    # comma_data_train = CommaLoader(path_npz_dummy,split_per, dummy_test= True, train= True)
    # train_loader = DataLoader(comma_data_train, batch_size=batch_size, shuffle=True)
    
    # comma_data_test = CommaLoader(path_npz_dummy,split_per, dummy_test= True, test=True)
    # test_loader = DataLoader(comma_data_test, batch_size=batch_size, shuffle=True)
    print("=>not using the dummy data rn")
elif "onnx" in name:
    comma_data_train = CommaLoader(path_comma_recordings, path_npz_dummy, 0.8, args.datatype,   train= True)
    train_loader = DataLoader(comma_data_train, batch_size = batch_size, shuffle= False, num_workers=n_workers)

    comma_data_val = CommaLoader(path_comma_recordings, path_npz_dummy, 0.8, args.datatype,  val= True)
    val_loader = DataLoader(comma_data_val, batch_size =  batch_size, shuffle= False, num_workers=n_workers)

##Load model 
"""
Both the model from scratch and the onnx-pytorch model can be used 

"""
#params for scratch model
inputs_dim_outputheads = {"path": 256, "ll_pred": 32, "llprob": 16, "road_edges": 16,
                          "lead_car": 64, "leadprob": 16, "desire_state": 32, "meta": [64, 32], "pose": 32}
output_dim_outputheads = {"path": 4955, "ll_pred": 132, "llprob": 8, "road_edges": 132,
                          "lead_car": 102, "leadprob": 3, "desire_state": 8, "meta": [48, 32], "pose": 12}
filters_list = [16, 24, 48, 88, 120, 208, 352]
expansion = 6
param_scratch_model = [filters_list, expansion, inputs_dim_outputheads,
                     output_dim_outputheads ]
pathplan_layer_names  = ["Gemm_959", "Gemm_981","Gemm_983","Gemm_1036"]

print("=>Loading the model")
print("=>model used:",args.modeltype)

def load_model(params_scratch, pathplan):

    if args.modeltype == 'scratch':
        model = CombinedModel(params_scratch[0], params_scratch[1],
                           params_scratch[2], params_scratch[3])
    else :
        onnx_model = onnx.load(onnx_path)
        model = ConvertModel(onnx_model)  #pretrained_model

        def reinitialise_weights(layer_weight):
            model.layer_weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(layer_weight))

        for name, layer in model.named_children():
            if isinstance(layer, torch.nn.Linear) and name in pathplan:
                reinitialise_weights(layer.weight)
                layer.bias.data.fill_(0.01)     
    return model 

comma_model = load_model(param_scratch_model, pathplan_layer_names)
comma_model = comma_model.to(device)
wandb.watch(comma_model) # Log the network weight histograms

#allowing grad only for path_plan
for name, param in comma_model.named_parameters():
    name_layer= name.split(".")
    if name_layer[0] in pathplan_layer_names:
        param.requires_grad = True
    else:
        param.requires_grad = False

## Define optimizer and scheduler
#diff. learning rate for different parts of the network.
if not diff_lr:
    param_group = comma_model.parameters()

    #optimizer for onnx and when diff_lr not True
    if args.modeltype == "onnx":
        optimizer = topt.Adam(param_group,lr[0], weight_decay=l2_lambda[0])
        scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lrs_factor, patience=lrs_patience, 
                                                 threshold=lrs_thresh, verbose=True, min_lr=lrs_min,
                                                 cooldown=lrs_cd)  
    else :
        pass # needs to be filled 

else:
    conv, gru, outhead = comma_model.getGroupParams()    
    gru = list(itertools.chain.from_iterable(gru))
    conv = list(itertools.chain.from_iterable(conv))
    outhead = list(itertools.chain.from_iterable(outhead))
    param_group = [ { "params": conv },
                    { "params": gru, "lr": lr[1], "weight_decay": l2_lambda[1] },
                    { "params": outhead, "lr": lr[2], "weight_decay": l2_lambda[2]}]
    
    optimizer = topt.Adam(param_group,lr[0], weight_decay=l2_lambda[0])
    #choice of lr_scheduler can be changed
    scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lrs_factor, patience=lrs_patience, 
                                                 threshold=lrs_thresh, verbose=True, min_lr=lrs_min,
                                                 cooldown=lrs_cd)                               

""""
        How to approach this situation :::
        1. create distribution objects and calcualte KL div. 
        2. class prob=== categorical distribution
        3. single logit === bernouolli distribution 
        4. for mean and std values -- normal distribution0
        5. for paths i need to take --- > argmin(losses) and add the loss in the final loss with that index
        6. for now use the most basic version of the pipeline. 
        7. for intial training give desire, traffic conv to zeros.
        8. How to scale the losses into scalar values for multidimensional kldiv values
        8. like for road edges and lanelines I have different branches so should i calculate loss for each of them separately
            and add them up or take them together as I am doing right now? 
        9. If multi-hypothesis exists it will be there for all the rest of the outputs with probabilities along with the output
            like plan, lanellines, road-edges and more... 
# """
#Loss functions:
def cal_path_loss(plan_pred, plan_gt, plan_prob_gt, batch_size ):
    ## path plan
    path_dict = {} 
    path_plans =  plan_pred
    path1, path2, path3, path4, path5 =torch.split(path_plans,991,dim=1)
    path_dict["path_prob"] = []
    path_dict["path1"] = path1[:,:-1].reshape(batch_size,2,33,15)
    path_dict["path2"] = path2[:,:-1].reshape(batch_size,2,33,15)
    path_dict["path3"] = path3[:,:-1].reshape(batch_size,2,33,15)
    path_dict["path4"] = path4[:,:-1].reshape(batch_size,2,33,15)
    path_dict["path5"] = path5[:,:-1].reshape(batch_size,2,33,15)
    path_pred_prob = torch.cat((path1[:,-1].reshape(batch_size,1), path2[:,-1].reshape(batch_size,1),path3[:,-1].reshape(batch_size,1),
                    path4[:,-1].reshape(batch_size,1),path5[:,-1].reshape(batch_size,1)),dim =1).reshape(batch_size,5,1)
    #naive path_loss---> train all the paths together

    path1_gt = plan_gt[:,0,:,:,:]
    path2_gt = plan_gt[:,1,:,:,:]
    path3_gt = plan_gt[:,2,:,:,:]
    path4_gt = plan_gt[:,3,:,:,:]
    path5_gt =plan_gt[:,4,:,:,:]
    
    def mean_std(array):
        mean = array[:,0,:,:]
        std = array[:,1,:,:]
        std = torch.exp(std) ## lower bound == 0; must be +ve
        return mean, std

    mean_pred_path1, std_pred_path1 = mean_std(path_dict["path1"])
    mean_gt_path1 , std_gt_path1 =  mean_std(path1_gt)

    mean_pred_path2, std_pred_path2 = mean_std(path_dict["path2"])
    mean_gt_path2 , std_gt_path2 =  mean_std(path2_gt)

    mean_pred_path3, std_pred_path3 = mean_std(path_dict["path3"])
    mean_gt_path3 , std_gt_path3 =  mean_std(path3_gt)
    
    mean_pred_path4, std_pred_path4 = mean_std(path_dict["path4"])
    mean_gt_path4 , std_gt_path4 =  mean_std(path4_gt)

    mean_pred_path5, std_pred_path5 = mean_std(path_dict["path5"])
    mean_gt_path5 , std_gt_path5 =  mean_std(path5_gt)
    
    
    def calcualte_path_loss(mean1, mean2, std1, std2):
        d1 = torch.distributions.normal.Normal(mean1, std1)
        d2 = torch.distributions.normal.Normal(mean2, std2)
        loss = torch.distributions.kl.kl_divergence(d1, d2).sum(dim =2).sum(dim =1).mean(dim =0)
        return loss

    path1_loss = calcualte_path_loss(mean_pred_path1, mean_gt_path1, std_pred_path1, std_gt_path1)
    path2_loss = calcualte_path_loss(mean_pred_path2, mean_gt_path2, std_pred_path2, std_gt_path2)
    path3_loss = calcualte_path_loss(mean_pred_path3, mean_gt_path3, std_pred_path3, std_gt_path3)
    path4_loss = calcualte_path_loss(mean_pred_path4, mean_gt_path4, std_pred_path4, std_gt_path4)
    path5_loss = calcualte_path_loss(mean_pred_path5, mean_gt_path5, std_pred_path5, std_gt_path5)

    path_pred_prob_d = torch.distributions.bernoulli.Bernoulli(logits = path_pred_prob)
    path_gt_prob_d = torch.distributions.bernoulli.Bernoulli(logits = plan_prob_gt)
    path_prob_loss =  torch.distributions.kl.kl_divergence(path_pred_prob_d, path_gt_prob_d).sum(dim=1).mean(dim=0)

    path_plan_loss = path1_loss + path2_loss + path3_loss + path4_loss + path5_loss + path_prob_loss
    
    return path_plan_loss

"""
Note: other loss functions to be used when training with scratch 
"""
# ## lanelines
# lane_pred = lane_pred.reshape(batch_size, 4,2,33,2)
# mean_lane_pred = lane_pred[:,:,0,:,:] 
# mean_lane_gt =  lane_line_gt[:,:,0,:,:]
# std_lane_pred = lane_pred[:,:,1,:,:]
# std_lane_gt = lane_line_gt[:,:,0,:,:]

# lane_pred_d = torch.distributions.normal.Normal(mean_lane_pred, std_lane_pred)
# lane_gt_d = torch.distributions.normal.Normal(mean_lane_gt, std_lane_gt)
# lane_loss = torch.distributions.kl.kl_divergence(lane_pred_d, lane_gt_d).sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)

# ## lanelines probability
# lane_prob_pred = lane_prob_pred.reshape(batch_size,4,2)

# lane_prob_pred_d = torch.distributions.categorical.Categorical(logits = lane_prob_pred)
# lane_prob_gt_d = torch.distributions.categorical.Categorical(logits = lane_prob_gt)
# lane_prob_loss = torch.distributions.kl.kl_divergence(lane_prob_pred_d, lane_prob_gt_d).sum(dim=1).mean(dim=0)

# ## Road edges
# road_edges_pred = road_edges_pred.reshape(batch_size,2,2,33,2)

# mean_road_edges_pred = road_edges_pred[:,:,0,:,:]
# mean_road_edges_gt = road_edges_gt[:,:,0,:,:]

# std_road_edges_pred = road_edges_pred[:,:,0,:,:]
# std_road_edges_gt = road_edges_gt[:,:,0,:,:]
# edges_pred_d = torch.distributions.normal.Normal(mean_road_edges_pred, std_road_edges_pred)
# edges_gt_d = torch.distributions.normal.Normal(mean_road_edges_gt, std_road_edges_gt)
# road_edges_loss = torch.distributions.kl.kl_divergence(edges_pred_d, edges_gt_d).sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)

# ## Lead car
# leads_pred = leads_pred.reshape(batch_size, 2, 51)
# leads_pred_other = leads_pred[:,:, :48].reshape(batch_size,2,2,6,4)
# leads_pred_prob = leads_pred[:,:, 48:]

# mean_leads_pred_other = leads_pred_other[:,:,0,:,:]
# mean_leads_gt = leads_gt[:,:,0,:,:]

# std_leads_pred_other = leads_pred_other[:,:,1,:,:]
# std_leads_gt =leads_gt[:,:,1,:,:]

# d1 = torch.distributions.normal.Normal(mean_leads_pred_other, std_leads_pred_other)
# d2 = torch.distributions.normal.Normal(mean_leads_gt, std_leads_gt)
# leads_other_loss = torch.distributions.kl.kl_divergence(d1, d2).sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)

# d3 = torch.distributions.categorical.Categorical(logits = leads_pred_prob)
# d4 = torch.distributions.categorical.Categorical(logits = leads_prob_gt)
# leads_prob_loss = torch.distributions.kl.kl_divergence(d3, d4).sum(dim = 1).mean(dim =0)

# Leads_loss = leads_other_loss + leads_prob_loss

# # ## Lead Probabilities
# lead_prob_gt = lead_prob_gt[:,0,:]

# lead_prob_pred_d=  torch.distributions.categorical.Categorical(logits = lead_prob_pred)
# lead_prob_gt_d = torch.distributions.categorical.Categorical(logits = lead_prob_gt)
# lead_prob_loss = torch.distributions.kl.kl_divergence(lead_prob_pred_d, lead_prob_gt_d).mean(dim=0)

# ## Desire
# desire_gt = desire_gt[:,0,:]
# desire_pred_d = torch.distributions.categorical.Categorical(logits = desire_pred)
# desire_gt_d = torch.distributions.categorical.Categorical(logits = desire_gt)
# desire_loss = torch.distributions.kl.kl_divergence(desire_pred_d, desire_gt_d).mean(dim=0)

# ## meta1 
# meta_pred_engagement = meta_pred[:,0].reshape(batch_size,1)
# meta_pred_various = meta_pred[:,1:36].reshape(batch_size, 5, 7)
# meta_pred_blinkers = meta_pred[:,36:].reshape(batch_size, 6, 2)
# meta_pred_engagement_d = torch.distributions.bernoulli.Bernoulli(logits = meta_pred_engagement)
# meta_gt_engagement_d = torch.distributions.bernoulli.Bernoulli(logits = meta_eng_gt)
# meta_engagement_loss = torch.distributions.kl.kl_divergence(meta_pred_engagement_d, meta_gt_engagement_d).mean(dim=0)

# meta_pred_various_d = torch.distributions.categorical.Categorical(logits = meta_pred_various)
# meta_gt_various_d = torch.distributions.categorical.Categorical(logits = meta_various_gt)
# meta_various_loss = torch.distributions.kl.kl_divergence(meta_pred_various_d, meta_gt_various_d).sum(dim =1).mean(dim =0)

# meta_pred_blinkers_d = torch.distributions.categorical.Categorical(logits = meta_pred_blinkers)
# meta_gt_blinkers_d = torch.distributions.categorical.Categorical(logits = meta_blinkers_gt)
# meta_blinkers_loss = torch.distributions.kl.kl_divergence(meta_pred_blinkers_d, meta_gt_blinkers_d).sum(dim =1).mean(dim =0)
        
# meta1loss = meta_engagement_loss + meta_various_loss + meta_blinkers_loss

# ## meta desire
# meta_desire_pred = meta_desire_pred.reshape(batch_size,4,8)
# meta_desire_pred_d =  torch.distributions.categorical.Categorical(logits = meta_desire_pred)
# meta_desire_gt_d = torch.distributions.categorical.Categorical(logits = meta_desire_gt)
# meta_desire_loss = torch.distributions.kl.kl_divergence(meta_desirlabels_path_probe_pred_d, meta_desire_gt_d).sum(dim=1).mean(dim =0)

# ##pose
# pose_pred = pose_pred.reshape(batch_size,2,6)        
# mean_pose_pred = pose_pred[:,0,:]
# std_pose_pred = pose_pred[:,1,:]
# mean_pose_gt = pose_gt[:,0,:]
# std_pose_gt = pose_gt[:,1,:]
# pose_pred_dist_obj = torch.distributions.normal.Normal(mean_pose_pred, std_pose_pred)
# pose_gt_dist_obj =  torch.distributions.normal.Normal(mean_pose_gt, std_pose_gt)
# pose_loss = torch.distributions.kl.kl_divergence(pose_pred_dist_obj, pose_gt_dist_obj)
# pose_loss = pose_loss.sum(dim=1).mean(dim=0)

### train loop 

#batchsize in this case is 1 as init recurr, desire and traffic conv reamain same. 
#initializing recurrent state by zeros
recurr_state = torch.zeros(1,512,dtype = torch.float32)
recurr_state = recurr_state.requires_grad_(True).to(device)

# desire and traffic convention is also set to zero
desire = torch.zeros(1,8,dtype = torch.float32, requires_grad= True)
desire = desire.requires_grad_(True).to(device)

#LHD
traffic_convention = torch.zeros(1,2, dtype = torch.float32)
traffic_convention[0][1] =1 
traffic_convention = traffic_convention.requires_grad_(True).to(device)


with run:
    wandb.config.lr = lr
    wandb.config.l2 = l2_lambda
    wandb.config.lrs = str(scheduler)
    wandb.config.seed = seed   
    for epoch in tqdm(range(epochs)):

        start_point = time.time()
        tr_loss = 0.0
        run_loss = 0.0

        for tr_it , data in tqdm(enumerate(train_loader)):
            if args.datatype == "dummy" and args.modeltype == "scratch":        
                #input
                yuv_images = input[0].to(device)
                desire = input[1].to(device)
                traffic_convention = input[2].to(device) 
                
                #gt
                plan_gt = labels[0].to(device)
                plan_prob_gt = labels[1].to(device)
                lane_line_gt = labels[2].to(device)
                lane_prob_gt = labels[3].to(device)
                road_edges_gt = labels[4].to(device) 
                leads_gt = labels[5].to(device)
                leads_prob_gt = labels[6].to(device)
                lead_prob_gt = labels[7].to(device)
                desire_gt = labels[8].to(device)
                meta_eng_gt = labels[9].to(device)
                meta_various_gt = labels[10].to(device)
                meta_blinkers_gt = labels[11].to(device)
                meta_desire_gt = labels[12].to(device)
                pose_gt = labels[13].to(device)
                desire = torch.squeeze(desire,dim =1)
                traffic_convention = torch.squeeze(traffic_convention, dim = 1)
        
                output1, output2 = comma_model(yuv_images, desire, recurrent_state, traffic_convention)
                plan_pred, lane_pred, lane_prob_pred, road_edges_pred, leads_pred, lead_prob_pred, desire_pred, meta_pred, meta_desire_pred, pose_pred = output1
            
                recurrent_state = output2 ## Feed back the recurrent state
                
            """
            add recurr state warmup:skip backward for some iterations.
            and move on to processing seq of batches.
            """
            if args.datatype == "gen_gt" and args.modeltype == "onnx":
                print("yes i am in the loop")    
                input, labels = data
                input = input.to(device)
                labels_path = labels[0].to(device)
                labels_path_prob = labels[1].to(device)
                optimizer.zero_grad()
                
                batch_loss = torch.zeros(1,dtype = torch.float32, requires_grad = True)
                
                # def updt_recurr(value_recurr):
                    
                #     update


                #     return update_recurr

                for i in range(batch_size):
                # recurr_state = recurrent_state.clone()
                
                    inputs_to_pretained_model = {"input_imgs":input[i],
                                                "desire": desire,
                                                "traffic_convention":traffic_convention,
                                                "initial_state": recurr_state}
                    
                    outputs = comma_model(**inputs_to_pretained_model) 
                    plan_predictions = outputs[:,:4955].clone()
                    recurr = outputs[:,5960:].clone() ## important to refeed state of GRU

                    single_itr_loss = cal_path_loss(plan_predictions, labels_path[i], labels_path_prob[i], 1)
                    if i == batch_size -1:
                        recurr = recurr
                    else:   
                        recurr_state = recurr

                    batch_loss += single_itr_loss
                batch_loss  = batch_loss/batch_size # mean of losses of samples in batch
                
                # recurrent warmup
                if recurr_warmup and epoch == 0 and tr_it>10:
                    batch_loss = batch_loss 
                else: 
                    batch_loss.backward(retain_graph = True)
                loss_cpu = batch_loss.detach().clone().item() ## this is the loss for one batch in one interation
                
                recurr_state = recurr
                tr_loss += loss_cpu
                run_loss += loss_cpu
                optimizer.step()
                
                if (tr_it+1)%10 == 0:
                    print("printing the losses")
                    print(f'{epoch+1}/{epochs}, step [{tr_it+1}/{len(train_loader)}], loss: {tr_loss/(tr_it+1):.4f}')
                    if (tr_it+1) %100 == 0:
                        # tr_logger.plotTr( run_loss /100, optimizer.param_groups[0]['lr'], time.time() - start_point ) ## add get current learning rate adjusted by the scheduler.
                        scheduler.step(run_loss/100)
                        run_loss =0.0
                    
            # validation loop  
                with torch.no_grad(): ## saving memory by not accumulating activations
                    if (epoch +1) %check_val_epoch ==0:
                        val_st_pt = time.time()
                        val_loss_cpu = 0.0
                        checkpoint_save_path = "./nets/checkpoints/commaitr" + date_it
                        torch.save(comma_model.state_dict(), checkpoint_save_path + (str(epoch+1) + ".pth" ))    
                        print(">>>>>validating<<<<<<<")

                        for val_itr, val_data in enumerate(val_loader):
                            val_input,val_labels = val_data

                            val_input = val_input.to(device)
                            val_label_path = val_labels[0].to(device)
                            val_label_path_prob = val_labels[1].to(device)
                            val_batch_loss = torch.zeros(1,dtype = torch.float32, requires_grad = True)
                            
                            for i in range(batch_size):
                                val_inputs_to_pretained_model = {"input_imgs":val_input[i],
                                                        "desire": desire,
                                                        "traffic_convention":traffic_convention,
                                                        "initial_state": recurr_state}
                            
                                val_outputs = comma_model(**val_inputs_to_pretained_model)
                                val_path_prediction = val_outputs[:,:4955].clone()

                                single_val_loss = cal_path_loss(val_path_prediction,val_label_path[i], val_label_path_prob[i], 1)
                                val_batch_loss += single_val_loss
                            
                            val_batch_loss = val_batch_loss/batch_size
                            val_loss_cpu += val_batch_loss.deatch().clone().cpu().item()

                            if (val_itr+1)%10 == 0:
                                print(f'Epoch:{epoch+1} ,step [{val_itr+1}/{len(val_loader)}], loss: {val_loss_cpu/(val_itr+1):.4f}')

                        print(f"Epoch: {epoch+1}, Val Loss: {val_loss_cpu/(len(val_loader)):.4f}")
                        val_logger.plotTr(val_loss_cpu, optimizer.param_groups[0]['lr'], time.time() - val_st_pt)
                            
        print(f"Epoch: {epoch+1}, Train Loss: {tr_loss/len(train_loader)}")
        tr_logger.plotTr(tr_loss/len(train_loader), optimizer.param_groups[0]['lr'], time.time() - start_point )

PATH = "./nets/model_itr/" +name + ".pth" 
torch.save(comma_model.state_dict(), PATH)
print( "Saved trained model" )