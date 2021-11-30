import glob
import onnxruntime as ort
from lanes_leads_gt import generate_ground_truth
import os


text_files = glob.glob("/gpfs/space/home/abdumali/comma2k19/**/video.hevc", recursive = True)
supercombo = ort.InferenceSession('supercombo.onnx')



for file in text_files:
    main_split = file.split( '|' )
    print(main_split)
    
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
