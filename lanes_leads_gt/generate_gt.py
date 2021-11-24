import glob
from lanes_leads_gt import generate_ground_truth

text_files = glob.glob("/data/realdata/aba20ae4/**/fcamera.hevc", recursive = True)

for file in text_files:
    print( "In processing ...", file )
    generate_ground_truth( file )