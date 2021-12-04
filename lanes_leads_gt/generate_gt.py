import glob
import onnxruntime as ort
from lanes_leads_gt import generate_ground_truth
import os



options = ort.SessionOptions() 
options.intra_op_num_threads = 10 
#options.inter_op_num_threads = 1

supercombo = ort.InferenceSession('supercombo.onnx', providers=["CUDAExecutionProvider"], sess_options=options)
text_files = glob.glob("/gpfs/space/projects/Bolt/comma_recordings/comma2k19/**/video.hevc", recursive = True)

for file in text_files:
    #main_split = file.split( '|' )
    #print(main_split)
    
    #if main_split:
        #print('been')
        #splits = file.split('/')
        #file2 = '/'.join( splits[:-2] )

        #newfolder = file2.replace('|', '_')
   
        #os.rename( file2, newfolder )
        #file = newfolder + '/'.join(splits[-2:])
    
    print( "In processing ...", file )
    
    #tdy:
    generate_ground_truth( file, supercombo )
    #except:
    # 	print( "Exception" )
