import numpy as np
from numpy import savez_compressed
from numpy import load

"""
To do: --- need to add the samples dimension.
"""
frame_length = 20

# input
def generate_input_arrays(frame_length):

    images = np.random.rand(frame_length, 12, 128, 256)
    desire = np.random.rand(frame_length,1, 8)
    traffic_convention = np.random.rand(frame_length,1, 2)  # assumed  LHD
    recurrent_state = np.random.rand(frame_length, 1, 512)  # needed on intialization

    return images, desire, traffic_convention, recurrent_state

def generate_gt_arrays(frame_length):

    plan = np.random.rand(frame_length,5, 2, 33, 15)
    plan_prob = np.random.rand(frame_length,5,1)
    ll = np.random.rand(frame_length,4, 2, 33, 2)
    ll_prob = np.random.rand(frame_length,4, 2)
    road_edg = np.random.rand(frame_length,2, 2, 33, 2)
    leads = np.random.rand(frame_length,2, 2, 6, 4)
    leads_prob = np.random.rand(frame_length, 2, 3)
    lead_prob = np.random.rand(frame_length,1, 3)
    desire = np.random.rand(frame_length,1, 8)
    meta_eng  = np.random.rand(frame_length,1)
    meta_various = np.random.rand(frame_length,5, 7)
    meta_blinkers = np.random.rand(frame_length,6, 2)
    meta_desire_exec = np.random.rand(frame_length,4, 8)
    pose = np.random.rand(frame_length,2, 6)

    return plan, plan_prob, ll, ll_prob, road_edg, leads, leads_prob, lead_prob, desire, meta_eng , meta_various, meta_blinkers, meta_desire_exec, pose

# generating sequence of inputs and saving them
imgs = []
desire = []
traff_conv = []
recurrent_state = []

input_1, input_2, input_3, input_4 = generate_input_arrays(frame_length)
imgs.append(input_1)
desire.append(input_2)
traff_conv.append(input_3)
recurrent_state.append(input_4)

# np.savez_compressed('inputdata.npz', 
#                     imgs = np.stack(imgs), 
#                     desire = np.stack(desire), 
#                     traff_conv = np.stack(traff_conv),
#                     recurrent_state = np.stack(recurrent_state))

# generating sequence of gt and saving them
plan = []
plan_prob = []
ll = []
ll_prob = []
road_edges = []
leads = []
leads_prob = []
lead_prob = []
desire = []
meta_eng = []
meta_various = []
meta_blinkers = []
meta_desire = []
pose = []

gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8, gt9, gt10, gt11, gt12, gt13, gt14 = generate_gt_arrays(frame_length)

plan.append(gt1)
plan_prob.append(gt2)
ll.append(gt3)
ll_prob.append(gt4)
road_edges.append(gt5)
leads.append(gt6)
leads_prob.append(gt7)
lead_prob.append(gt8)
desire.append(gt9)
meta_eng.append(gt10)
meta_various.append(gt11)
meta_blinkers.append(gt12)
meta_desire.append(gt13)
pose.append(gt14)

# np.savez_compressed('gtdata.npz',
#                     plan = plan,
#                     plan_prob = plan_prob,
#                     ll = ll,
#                     ll_prob = ll_prob,
#                     road_edges = road_edges, 
#                     leads = leads, 
#                     leads_prob = leads_prob,
#                     lead_prob = lead_prob,
#                     desire = desire, 
#                     meta_eng = meta_eng,
#                     meta_various = meta_various, 
#                     meta_blinkers = meta_blinkers,
#                     meta_desire = meta_desire, 
#                     pose = pose)

# data = load("gtdata.npz")

# for k in data.iterkeys():
#     print(k)

# # # correct way to find the length of this dummy data
# a = data['meta_eng'][0]
# print(a.shape) 

# # a = data["imgs"][0][:16]
# splt = 16
# for i in range(4):
#     print(data["imgs"][0][16+i].shape)
#     # print(16+i)

