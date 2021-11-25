import glob
import onnxruntime as ort

from lanes_leads_gt import generate_ground_truth

text_files = glob.glob("/data/realdata/aba20ae4/**/fcamera.hevc", recursive = True)

supercombo = ort.InferenceSession('supercombo.onnx')

for file in text_files:
    print( "In processing ...", file )
    try:
        generate_ground_truth( file, supercombo )
    except:
        print( "Exception" )

print( text_files )