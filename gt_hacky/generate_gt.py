import glob
import onnxruntime as ort
from lanes_leads_gt import generate_ground_truth
import os


if __name__ == '__main__':

    options = ort.SessionOptions() 
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    supercombo = ort.InferenceSession('../common/supercombo.onnx', providers=["CUDAExecutionProvider"], sess_options=options)
    segment_videos = glob.glob("/gpfs/space/projects/Bolt/comma_recordings/comma2k19/**/video.hevc", recursive = True)

    for path_to_video in segment_videos:   
        print( "In processing ...", path_to_video )
    
        generate_ground_truth(path_to_video, supercombo)
