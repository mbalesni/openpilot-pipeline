# openpilot-pipeline

Comma AI's Advanced Driver-Assitance System (ADAS) [Openpilot](https://github.com/commaai/openpilot) is open-source, but the code to create training data and train the models is not. We attempt to re-create & open-source their full pipeline.

## About @nikebless
- where this project is right now? (distillation only, no true training)

## Model @gauti

- inputs
- outputs
- technicalities: onnx2pytorch @nikebless
- loss functions

## Data pipeline @nikebless
- gt_hacky — distilling existing model
- gt_real (not finished yet)

## Training pipeline

### Training loop @gauti
- GRU training logic (stopping gradients, recurrent warmup, resetting after each segment)
- Visualization of predictions vs GTs
- Wandb

### Data loading @nikebless
- How we create batches (mention requirement of 1 CPU per 1 sample)

## How to Use

### System Requirements @nikebless
- for GT creation we need sudo + probably Ubuntu 20.04 (for compiling openpilot)
- for training, no sudo required, anything that can run pytorch

### Installations @gauti

1. Install openpilot (for LogReader mainly)
2. Install environment

### Running

1. Get dataset: @nikebless
  - simplest: comma2k19
  - custom data small scale: copy from comma device
  - custom data large scale: retropilot
2. Run ground truth creation @nikebless
  - gt_hacky + parse_logs (mention we're not using RPY yet)
3. Set up wandb @gauti
4. Run training @gauti

### Using the model @gauti


0. Save model in ONNX
1. In simulation (Carla)
2. In the Comma device — Convert to DLC @rabdumalikov

## Our Results @gauti

- (maybe) loss metrics
- visualized predictions

## Technical improvement ToDos

- [ ] Do not crash training when a single video failed to read