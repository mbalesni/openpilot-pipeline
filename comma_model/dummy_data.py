import numpy as np
from numpy import savez_compressed
from numpy import load

rand_seq_length = 10

# input
def generate_input_arrays():
    images = np.random.rand(1, 12, 128, 256)
    desire = np.random.rand(1, 8)
    traffic_convention = np.random.rand(1, 2)  # assumed  LHD
    recurrent_state = np.random.rand(1, 512)  # needed on intialization

    return images, desire, traffic_convention, recurrent_state


def generate_gt_arrays():

    plan = np.random.rand(5, 2, 33, 15)
    ll = np.random.rand(4, 2, 33, 2)
    ll_prob = np.random.rand(4, 2)
    road_edg = np.random.rand(2, 2, 33, 2)
    leads = np.random.rand(2, 2, 6, 4)
    leads_prob = np.random.rand(1, 3)
    desire = np.random.rand(1, 8)
    meta_various = np.random.rand(1, 48)  # (1, 5*7, 6*2)
    meta_desire_exec = np.random.rand(4, 8)
    pose = np.random.rand(2, 6)

    return plan, ll, ll_prob, road_edg, leads, leads_prob, desire, meta_various, meta_desire_exec, pose


# generating sequence of inputs and saving them
imgs = []
desire = []
traff_conv = []
recurrent_state = []

for i in range(rand_seq_length):
    input_1, input_2, input_3, input_4 = generate_input_arrays()
    imgs.append(input_1)
    desire.append(input_2)
    traff_conv.append(input_3)

    if i == 0:
        recurrent_state.append(input_4)

np.savez_compressed('inputdata.npz', imgs, desire, traff_conv, recurrent_state)

# generating sequence of gt and saving them
plan = []
ll = []
ll_prob = []
road_edges = []
leads = []
leads_prob = []
desire = []
meta_various = []
meta_desire = []
pose = []

for i in range(rand_seq_length):
    gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8, gt9, gt10 = generate_gt_arrays()

    plan.append(gt1)
    ll.append(gt2)
    ll_prob.append(gt3)
    road_edges.append(gt4)
    leads.append(gt5)
    leads_prob.append(gt6)
    desire.append(gt7)
    meta_various.append(gt8)
    meta_desire.append(gt9)
    pose.append(gt10)

np.savez_compressed('gtdata.npz', plan, ll, ll_prob, road_edges, leads, leads_prob,
                    desire, meta_various, meta_desire, pose)

# data = load("gtdata.npz")
# for k in data.iterkeys():
#     print(k)
# print(data['arr_0'][0].shape)
