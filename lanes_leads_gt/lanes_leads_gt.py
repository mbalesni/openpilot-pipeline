#!/usr/bin/env python3

import sys
sys.path.append("../")


from utils import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import onnxruntime as ort

import cv2 
from tensorflow.keras.models import load_model

MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10


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


def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

def generate_ground_truth( camerafile ):
  splits = camerafile.split('/')
  path_to_video_file = '/'.join(splits[:-1])

  supercombo = ort.InferenceSession('supercombo.onnx')

  cap = cv2.VideoCapture(camerafile)

  imgs = []

  for i in tqdm(range(1000)):
    ret, frame = cap.read()
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    imgs.append(img_yuv.reshape((874*3//2, 1164)))
  

  imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
  for i, img in tqdm(enumerate(imgs)):
    imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                      output_size=(512,256))
  frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0


  state = np.zeros((1,512)).astype( np.float32 )
  desire = np.zeros((1,8)).astype( np.float32 )

  cap = cv2.VideoCapture(camerafile)

  tc = np.array([[0,1]]).astype( np.float32 )

  plans = []
  lanelines = []
  laneline_probs = []
  road_edges = []
  leads = []
  lead_probs = []
  desire_out = []
  metas = [] 
  poses = []

  for i in tqdm(range(len(frame_tensors) - 1)):
    
    img = np.vstack(frame_tensors[i:i+2])[None]
    outs = supercombo.run( None, { 'input_imgs': img, 'desire': desire, 'traffic_convention': tc, 'initial_state': state } )

    outs = outs[0][0]
    
    plans.append( np.reshape( outs[:PATH], ( 5, 991 ) ) )
    lanelines.append( np.reshape( outs[PATH:LANE_LINES], (4,132) ) )
    laneline_probs.append( np.reshape( outs[LANE_LINES:LANE_LINE_PROB], (4,2) ) )
    road_edges.append( np.reshape( outs[LANE_LINE_PROB:ROAD_EDGES], (2,132) ) )
    leads.append( np.reshape( outs[ROAD_EDGES:LEADS], (2,51) ) )
    lead_probs.append( np.reshape( outs[LEADS:LEAD_PROB], (1,3) ) )
    desire_out.append( outs[LEAD_PROB:DESIRE_STATE] )
    metas.append( outs[DESIRE_STATE:META] )
    poses.append( np.reshape( outs[META:POSE], (2,6) ) )

    recurrent_state = outs[POSE:RECURRENT_STATE] 

    # Important to refeed the state
    state = [recurrent_state]

  plans = np.stack( plans )
  lanelines = np.stack( lanelines )
  laneline_probs = np.stack( laneline_probs )
  road_edges = np.stack( road_edges )
  leads = np.stack( leads )
  lead_probs = np.stack( lead_probs )
  desire_out = np.stack( desire_out )
  metas = np.stack( metas )
  poses = np.stack( poses )

  np.savez_compressed(path_to_video_file + '/marker_and_leads_ground_truth.npz',
    plan=plans,
    lanelines=lanelines,
    laneline_probs=laneline_probs,
    road_edges=road_edges,
    leads=leads,
    lead_probs=lead_probs,
    desire=desire_out,
    meta=metas,
    pose=poses )


if __name__ == '__main__':
  camerafile = sys.argv[1]

  generate_ground_truth( camerafile )
