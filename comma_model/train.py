import torch 
import argparse
import torch.nn as nn
from torchvision.transforms import functional as F
import os 
import cv2 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## CLI parser 
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default = "", help= "directory in which the dummy data or generated gt lies" )
parser.add_argument("--batch_size", type= int, default=1, help = "batch size")
parser.add_argument("--num_gpu", type=int, default =1, help= "number of gpus")
parser.add_argument("--losstype", type= str, default= "KL_div", help= "choose between Kl and NNL loss")

#hyperparams
epochs = 20 
learning_rate = 0.001
batch_size = 2 

### Load data 




##Load model 



## Define Loss 




## train loop 