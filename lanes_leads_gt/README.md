Return the ground-truth for a road segment.

## USAGE 1

**Script:** start.sh

**Input:** fcamera.hvec

**Output:** marker_and_leads_ground_truth.npz

**Output format:** [ plan: [...], lanelines: [...], leads: [...], ..., pose: [...] ]

**Example:** ./start.sh fcamera.hvec

## USAGE 2

**Script:** generate_gt.py

**Description:** generate ground truth in-place for all the folders in '/data/realdata/aba20ae4' which contains 'fcamera.hevc'.

**Example:** python generate_gt.py

## INSTALL 

You can use install.sh script and if it won't work try to invoke all the commands
from this file one by one.


