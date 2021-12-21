cp lanes_leads_gt.py modeld/
cd modeld/
python3.8 lanes_leads_gt.py $1
cp marker_and_leads_ground_truth.npz ../marker_and_leads_ground_truth.npz