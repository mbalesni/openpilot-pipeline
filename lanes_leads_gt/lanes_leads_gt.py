#!/usr/bin/env python3
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import onnxruntime as ort

import cv2 
from tensorflow.keras.models import load_model
from common.tools.lib.parser import parser
import sys
camerafile = sys.argv[1]
supercombo = ort.InferenceSession('models/supercombo.onnx')

MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

cap = cv2.VideoCapture(camerafile)

imgs = []

for i in tqdm(range(1000)):
  ret, frame = cap.read()
  img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  imgs.append(img_yuv.reshape((874*3//2, 1164)))
 

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

imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
for i, img in tqdm(enumerate(imgs)):
  imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0


state = np.zeros((1,512)).astype( np.float32 )
desire = np.zeros((1,8)).astype( np.float32 )

cap = cv2.VideoCapture(camerafile)

tc = np.array([[0,1]]).astype( np.float32 )

ground_truth = []

for i in tqdm(range(len(frame_tensors) - 1)):
  
  img = np.vstack(frame_tensors[i:i+2])[None]
  outs = supercombo.run( None, { 'input_imgs': img, 'desire': desire, 'traffic_convention': tc, 'initial_state': state } )

  outs = outs[0][0]

  # Important to refeed the state
  state = [outs[4955+528+8+264+102+3+8+80+12:4955+528+8+264+102+3+8+80+12+512]]

  lanelines = outs[4955:4955+528]
  leads = outs[4955+528+8+264:4955+528+8+264+102]

  ground_truth.append( { 'lanelines': lanelines, 'leads': leads } )
  
np.savez_compressed('marker_and_leads_ground_truth.npz',
  gt=ground_truth )
