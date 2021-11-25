import argparse
import itertools
import os 
import cv2
import numpy as np 
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

"""
dataset split for test and train will create a new utils.py for that later
"""

def split_data():



if "Dummy" or "dummy" in name:
    dummy_data  = CommaLoader(path_npz_dummy, dummy_test= True)
    dummy_data_loader = DataLoader(dummy_data, batch_size=2, shuffle=True)

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
# loss_func = 

# ## train loop 




