# Creating ground truths for Distillation

Create ground truth predictions by running an existing model on driving data in the comma2k19 format.

## Usage

```bash
python generate_gt.py /path/to/segments
```

Generates ground truth files in-place for all nested folders under <path_to_segments> that contain files `fcamera.hevc` or `video.hevc`.



