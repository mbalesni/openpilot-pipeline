import argparse
import itertools
import os 
import cv2
import numpy as np 
from tqdm import tqdm
import torch 
import torch.nn as nn
from torchvision.transforms import functional as F
import torch.optim as topt
from Dataloader import *
from model import *

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
print("seed={}".format(seed))

# CLI parser 
parser = argparse.ArgumentParser(description='Args for comma supercombo train pipeline')
# parser.add_argument("--datadir", type=str, default = "", choices=["dummy", "gen_gt"], help= "directory in which the dummy data or generated gt lies" )
# parser.add_argument("--num_gpu", type=int, default =1, help= "number of gpus")
# parser.add_argument("--mode", type =str, default="train", choices = ["train", "val", "eval"])
# parser.add_argument("--batch_size", type= int, default=1, help = "batch size")
# parser.add_argument("--losstype", type= str, default= "KL_div", help= "choose between Kl and NNL loss")
parser.add_argument("--modeltype", type = str, default = "scratch", choices= ["scratch", "onnx"], help = "choose type of model for train")

args = parser.parse_args()
# print(args.modeltype)

#Hyperparams
name = "Dummy_comma_pipeline_nov26"
path_npz_dummy = ["inputdata.npz","gtdata.npz"] # dummy data_path
lr = (1e-4, 2e-4, 1e-3) ## (lr_conv, lr_gru, lr_outhead)
diff_lr = True
l2_lambda = (1e-4,1e-4,1e-4) 
lrs_factor = 0.75
lrs_patience = 50
lrs_cd = 50
lrs_thresh = 1e-4
lrs_min = 1e-6

epochs = 20 
batch_size = 2
split_per = 0.8

### Load data and split in test and train

if "Dummy" or "dummy" in name:
    comma_data_train = CommaLoader(path_npz_dummy,split_per, dummy_test= True, train= True)
    train_loader = DataLoader(comma_data_train, batch_size=batch_size, shuffle=True)
    
    comma_data_test = CommaLoader(path_npz_dummy,split_per, dummy_test= True, test=True)
    test_loader = DataLoader(comma_data_test, batch_size=batch_size, shuffle=True)
    
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


def load_model(params_scratch):
    if args.modeltype == 'scratch':
        model = CombinedModel(params_scratch[0], params_scratch[1],
                           params_scratch[2], params_scratch[3])
    else : 
        pass ## onnx-pytorch model 
    
    return model 

comma_model = load_model(param_scratch_model)
comma_model = comma_model.to(device)

### Define Loss and optimizer
#diff. learning rate for different parts of the network.
if not diff_lr:
    param_group = comma_model.parameters()
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
criterion1 = nn.KLDivLoss()
criterion2 = nn.CrossEntropyLoss()
sftmax = nn.Softmax(dim=0)

# ## train loop 
recurrent_state = torch.zeros(batch_size,512)
for epoch in tqdm(range(epochs)):
    for i , data in tqdm(enumerate(train_loader)):
        input, labels = data
        optimizer.zero_grad()
        #input
        yuv_images = input[0].to(device)
        desire = input[1].to(device)
        traffic_convention = input[2].to(device) 
        
        #gt
        plan_gt = labels[0].to(device)
        lane_line_gt = labels[1].to(device)
        lane_prob_gt = labels[2].to(device)
        road_edges_gt = labels[3].to(device) 
        leads_gt = labels[4].to(device)
        lead_prob_gt = labels[5].to(device)
        desire_gt = labels[6].to(device)
        meta_gt = labels[7].to(device)
        meta_desire_gt = labels[8].to(device)
        pose_gt = labels[9].to(device)
        desire = torch.squeeze(desire,dim =1)
        traffic_convention = torch.squeeze(traffic_convention, dim = 1)
        output1, output2 = comma_model(yuv_images, desire, recurrent_state, traffic_convention)
        plan_pred, lane_pred, lane_prob_pred, road_edges_pred, leads_pred, lead_prob_pred, desire_pred, meta_pred, meta_desire_pred, pose_pred = output1
        
        recurrent_state = output2
        """"
        How to approach this situation :::
        1. create distribution objects and calcualte KL div. 
        2. class prob=== categorical distribution
        3. single logit === bernouolli distribution 
        4. for mean and std values -- normal distribution0
        5. for paths i need to take --- > argmin(losses) and add the loss in the final loss with that index
        6. for now use the most basic version of the pipeline. 
        7. for intial training give desire, traffic conv to zeros.
        """

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
        path_dict["path_prob"].append(path1[:,-1]) 
        path_dict["path_prob"].append(path2[:,-1])
        path_dict["path_prob"].append(path3[:,-1])
        path_dict["path_prob"].append(path4[:,-1])
        path_dict["path_prob"].append(path5[:,-1])
    
        # plan_hyp_tensor = torch.tensor((path_dict['path_prob'][0],path_dict['path_prob'][1],path_dict['path_prob'][2],path_dict['path_prob'][3],path_dict['path_prob'][4]), requires_grad= False )
        
        # # print(plan_hyp_tensor)
        # plan_hyp_tensor = sftmax(plan_hyp_tensor)
        # # print(plan_hyp_tensor)
        # plan_branch_index = torch.argmax(plan_hyp_tensor).item()

        ## lanelines



        ## lanelines probability 


        ## Road edges


        ## Lead car


        ## Lead Probabilities)



        ## Desire
        if "dummy" or "Dummy" in name:
            desire_gt = desire_gt[:,0,:]
        else :
            desire_gt = desire_gt 

            
        desire_pred_d = torch.distributions.categorical.Categorical(logits = desire_pred)
        desire_gt_d = torch.distributions.categorical.Categorical(logits = desire_gt)
        desire_loss = torch.distributions.kl.kl_divergence(desire_pred_d, desire_gt_d).mean(dim=0)

        ## meta1 
        if "dummy" or "Dummy" in name:
            meta_gt = meta_gt[:,0,:]
            meta_gt_engagement = meta_gt[:,0]
            meta_gt_various = meta_gt[:,1:36].reshape(batch_size, 5, 7)
            meta_gt_blinkers = meta_gt[:,36:].reshape(batch_size, 6, 2)
        else:
            pass # fill condition for real data

        
        meta_pred_engagement = meta_pred[:,0]
        meta_pred_various = meta_pred[:,1:36].reshape(batch_size, 5, 7)
        meta_pred_blinkers = meta_pred[:,36:].reshape(batch_size, 6, 2)

        meta_pred_engagement_d = torch.distributions.bernoulli.Bernoulli(logits = meta_pred_engagement)
        meta_gt_engagement_d = torch.distributions.bernoulli.Bernoulli(logits = meta_gt_engagement)
        meta_engagement_loss = torch.distributions.kl.kl_divergence(meta_pred_engagement_d, meta_gt_engagement_d).mean(dim=0)

        meta_pred_various_d = torch.distributions.categorical.Categorical(logits = meta_pred_various)
        meta_gt_various_d = torch.distributions.categorical.Categorical(logits = meta_gt_various)
        meat_various_loss = torch.distributions.kl.kl_divergence(meta_pred_various_d, meta_gt_various_d).sum(dim =1).mean(dim =0)

        meta_pred_blinkers_d = torch.distributions.categorical.Categorical(logits = meta_pred_blinkers)
        meta_gt_blinkers_d = torch.distributions.categorical.Categorical(logits = meta_gt_blinkers)
        meta_blinkers_loss = torch.distributions.kl.kl_divergence(meta_pred_blinkers_d, meta_gt_blinkers_d).sum(dim =1).mean(dim =0)
                
        meta1loss = meta_engagement_loss + meat_various_loss + meta_blinkers_loss

        ## meta desire
        meta_desire_pred = meta_desire_pred.reshape(batch_size,4,8)
        meta_desire_pred_d =  torch.distributions.categorical.Categorical(logits = meta_desire_pred)
        meta_desire_gt_d = torch.distributions.categorical.Categorical(logits = meta_desire_gt)
        meta_desire_loss = torch.distributions.kl.kl_divergence(meta_desire_pred_d, meta_desire_gt_d)
        meta_desire_loss = meta_desire_loss.sum(dim=1).mean(dim =0)
        
        ##pose
        pose_pred = pose_pred.reshape(batch_size,2,6)        
        mean_pose_pred = pose_pred[:,0,:]
        std_pose_pred = pose_pred[:,1,:]
        mean_pose_gt = pose_gt[:,0,:]
        std_pose_gt = pose_gt[:,1,:]
        pose_pred_dist_obj = torch.distributions.normal.Normal(mean_pose_pred, std_pose_pred)
        pose_gt_dist_obj =  torch.distributions.normal.Normal(mean_pose_gt, std_pose_gt)
        pose_loss = torch.distributions.kl.kl_divergence(pose_pred_dist_obj, pose_gt_dist_obj)
        pose_loss = pose_loss.sum(dim=1).mean(dim=0)

        ## task loss balancing strategy: used--> most naive
        # Combined_loss= 


        ## backprop 
        # Combined_loss.backward()

