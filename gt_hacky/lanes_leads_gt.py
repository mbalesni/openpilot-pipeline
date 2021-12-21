#!/usr/bin/env python3

import sys
sys.path.append("../")

from utils import get_train_imgs
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from os.path import exists

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

def generate_ground_truth( camerafile, supercombo ):
  splits = camerafile.split('/')
  path_to_video_file = '/'.join(splits[:-1])
  file_name = splits[-1]

  if exists(path_to_video_file + '/marker_and_leads_ground_truth.npz' ):
    print( "File already exist!" )
    return
  
  images = []
  try:  
    images = get_train_imgs( path_to_segment=path_to_video_file, video_file=file_name )
  except Exception as e:
    print( "can't read images" )
    print(e)
    return 

  state = np.zeros((1,512)).astype( np.float32 )
  desire = np.zeros((1,8)).astype( np.float32 )

  tc = np.array([[0,1]]).astype( np.float32 )

  plans = []
  plans_prob = []
  lanelines = []
  laneline_probs = []
  road_edges = []
  leads_pred = []
  leads_prob = []
  lead_probs = []
  desire_out = []
  meta_engage_prob = [] 
  meta_various_prob = [] 
  meta_blinkers_prob = [] 
  meta_desires = [] 
  poses = []

  for i in tqdm(range(len(images) - 1)):
    
    img = np.vstack(images[i:i+2])[None]
    outs = supercombo.run( None, { 'input_imgs': img, 'desire': desire, 'traffic_convention': tc, 'initial_state': state } )

    outs = outs[0][0]
    
    p = outs[:PATH]
    plans.append( np.reshape( p[:PATH-5], ( 5, 2, 33, 15 ) ) )
    plans_prob.append( np.reshape( p[PATH-5:PATH], ( 5, 1 ) ) )
    lanelines.append( np.reshape( outs[PATH:LANE_LINES], (4, 2, 33, 2) ) )
    laneline_probs.append( np.reshape( outs[LANE_LINES:LANE_LINE_PROB], (4,2) ) )
    road_edges.append( np.reshape( outs[LANE_LINE_PROB:ROAD_EDGES], (2,2,33,2) ) )

    l = outs[ROAD_EDGES:LEADS]
    leads_pred.append( np.reshape( l[:len(l)-6], (2,2,6,4) ) )
    leads_prob.append( np.reshape( l[len(l)-6:len(l)], (2,3) ) )

    lead_probs.append( np.reshape( outs[LEADS:LEAD_PROB], (1,3) ) )
    desire_out.append( outs[LEAD_PROB:DESIRE_STATE] )

    m = outs[DESIRE_STATE:META] 

    meta_offsets = [0, (1,36), (36, 48), (48,80) ]

    meta_engage_prob.append( m[meta_offsets[0]] )
    
    meta_various_prob.append( np.reshape( m[meta_offsets[1][0]:meta_offsets[1][1]], (5,7) ) )
    meta_blinkers_prob.append( np.reshape( m[meta_offsets[2][0]:meta_offsets[2][1]], (6,2) ) )
    meta_desires.append( np.reshape( m[meta_offsets[3][0]:meta_offsets[3][1]], (4,8) ) )

    poses.append( np.reshape( outs[META:POSE], (2,6) ) )

    recurrent_state = outs[POSE:RECURRENT_STATE] 

    # Important to refeed the state
    state = [recurrent_state]

  if not plans:
    return 

  try:
    np.savez_compressed(path_to_video_file + '/marker_and_leads_ground_truth.npz',
    plans=np.stack( plans ),
    plans_prob=np.stack( plans_prob ),
    lanelines=np.stack( lanelines ),
    laneline_probs=np.stack( laneline_probs ),
    road_edges=np.stack( road_edges ),
    leads_pred=np.stack( leads_pred ),
    leads_prob=np.stack( leads_prob ),
    lead_probs=np.stack( lead_probs ),
    desire=np.stack( desire_out ),
    meta_engage_prob=np.stack( meta_engage_prob ),
    meta_various_prob=np.stack( meta_various_prob ),
    meta_blinkers_prob=np.stack( meta_blinkers_prob ),
    meta_desires=np.stack( meta_desires ),
    pose=np.stack( poses ) )
  except:
    print( "Can't save npz file" )

if __name__ == '__main__':
  camerafile = sys.argv[1]

  supercombo = ort.InferenceSession('supercombo.onnx')

  generate_ground_truth( camerafile, supercombo )
