import numpy as np 
from numpy import savez_compressed
from numpy import load 

rand_seq_length = 36
### input 
def generate_input_arrays():
    images = np.random.rand(1,12,128,256)
    desire = np.random.rand(1,8)
    traffic_convention = np.random.rand(1,2) # assumed  LHD
    recurrent_state = np.random.rand(1,512) # needed on intialization

    return images, desire, traffic_convention, recurrent_state 


def generate_gt_arrays():
    
    plan = np.random.rand(5,2,33,15)
    ll = np.random.rand(4,2,33,2)
    ll_prob = np.random.rand(4,2)
    road_edg = np.random.rand(2,2,33,2)
    leads = np.random.rand(2,2,6,4)
    leads_prob = np.random.rand(1,3)
    desire = np.random.rand(1,8)
    meta_various  = np.random.rand(1,48) ## (1, 5*7, 6*2)
    meta_desire_exec = np.random.rand(4,8)
    pose = np.random.rand(2,6)
    
    return plan, ll, ll_prob, road_edg, leads, leads_prob, desire, meta_various, meta_desire_exec, pose  


#generating sequence of inputs and saving them
# for i in range(rand_seq_length):
    
    



# np.savez_compressed('inputdata.npz',a)
# data = load("inputdata.npz")
# for k in data.iterkeys():
#     print(k)
