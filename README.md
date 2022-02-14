# openpilot-pipeline

[Openpilot](https://github.com/commaai/openpilot) is the currently leading[^1] Advanced Driver-Assitance System (ADAS), developed & open-sourced by [Comma AI](https://comma.ai/).

This repo attempts to re-create the complete data & training pipeline to allow training custom driving models for Openpilot.

![compressed-model](https://user-images.githubusercontent.com/25569111/151782155-59a3fe8f-e12e-414b-9f58-1c699924eb1c.gif)


## About

The project is *in early development*. The ultimate goal is to create a codebase for training end-to-end autonomous driving models.

Right now, the only implemented feature is distillation of path planning *from the original Openpilot driving model*. To train from scratch, we need to implement the creation of accurate ground truth paths by processing data from cars' GNSS, IMU, and visual odometry with [laika](https://github.com/commaai/laika) and [rednose](https://github.com/commaai/rednose) (PRs welcome!).

Below we describe the implemented parts of the pipeline and current training results.

## Model
The neural network architecture can be broken down into three parts:
convolutional feature extractor (based on Resnet), followed by
a GRU (used to capture the temporal context)
several fully-connected branches for outputting paths, lane lines, road edges, etc. ( explained in detail below) 

<table>
  <tr>
    <td>CNN Feature Extractor</td>
     <td>GRU</td>
     <td>Output Heads</td>
  </tr>
  <tr>
    <td><img src="doc/Conv_extractor.png" width=100 height=250></td>
    <td><img src="doc/GRU.png" width=300 height=250></td>
    <td><img src="doc/output_heads.png"  height=250></td>
  </tr>
 </table>

The model visualized above is `supercombo.onnx` from [Openpilot 0.8.10 Release](https://github.com/commaai/openpilot/tree/v0.8.10/models). You can find a copy of it in this [repository ](https://github.com/nikebless/openpilot-pipeline/tree/main/common/models). 

Detailed definitions of the inputs and outputs of the model are mentioned [here](https://github.com/commaai/openpilot/tree/master/models). 
### Inputs
As mentioned above, we can divide the model architecture into encoder and decoder. The encoder includes a CNN feature extractor and a GRU, and the decoder is several branches of fully connected layers ending with task-specific outputs. 

The primary input to the model is two consecutive camera frames. They are converted into YUV 4:2:0 format, reprojected, and stacked to form a tensor of shape `(N,12,128,256)`.
* ##### YUV 4:2:0
  * It is a color subsampling technique.
  * Encodes only 20% of the color infromation.
  * Saves memory footprint
  * Computationally efficient. 

Three other model inputs are fed after the convolutional extractor as input for the GRU: desire, traffic convention, and recurrent state. 

* ##### Desire
  Desire is a one-hot vector of shape `(N,8`). It is used to condition the model for specific actions:

    ![desire](doc/desire.png?raw=true)
    
  The class above is from the [logs classes definition file](https://github.com/commaai/cereal/blob/5c64eaab789dfa67a210deaa2530788843474999/log.capnp#L894-L902). 

* ##### Traffic Convention
  * Traffic convention is a one-hot vector of shape `(N,2)`, conditioning the model with the local driving side convention (left-hand or right-hand).

* ##### Recurrent State
  * GRU memory of shape `(N,512)`. 

### Outputs
The outputs from the task-specific branches are concatenated, resulting in the model output of shape `(N,6472)`.
The code to manually parse the outputs can be found in the older versions of the [openpilot](https://github.com/commaai/openpilot/blob/v0.8.9/selfdrive/modeld/models/driving.cc#L15-L49). Most outputs are predicted in the form of mean and standard deviation over time. Path plans, lane lines, road edges, etc are predicted for 33 timestamps [quadratically spaced out](https://github.com/commaai/openpilot/blob/7d3ad941bc4ba4c923af7a1d7b48544bfc0d3e13/selfdrive/common/modeldata.h#L14-L25) for 10 seconds or 192 meters from the current position. These predictions are in the so-called calibrated frame of reference, explained in detail [here](https://github.com/commaai/openpilot/tree/master/common/transformations).

Comma explain model outputs in the [Openpilot model readme](https://github.com/commaai/openpilot/tree/7d3ad941bc4ba4c923af7a1d7b48544bfc0d3e13/models).

### Converting the model from ONNX to PyTorch

To fine-tune an existing driving model, we convert it from the ONNX format to PyTorch using [onnx2pytorch](https://github.com/ToriML/onnx2pytorch). Note: there is a bug in onnx2pytorch that impacts the outputs; until this [PR](https://github.com/ToriML/onnx2pytorch/pull/38) is merged, a [manual fix](https://github.com/nikebless/openpilot-pipeline/blob/main/train/model.py#L28-L29) is necessary.

### Loss functions

As mentioned above, we are currently doing only distillation of path prediction. We implement two loss functions:

- [KL-divergence](https://github.com/nikebless/openpilot-pipeline/blob/main/train/train.py#L69-L77)
- [Winner-takes-all Laplacian NLL](https://github.com/nikebless/openpilot-pipeline/blob/main/train/train.py#L61-L66)

The latter one works much better, and the code for it was shared with us by folks at CommaAI (thanks a lot!).

## Data pipeline

A script in `gt_hacky` runs the official Openpilot model on the full dataset and saves the outputs.

True ground truth creation is currently not implemented.

For the dataset, we use [comma2k19](https://github.com/commaai/comma2k19), a 33-hour (1980 min) driving dataset made by CommaAI: 

> The data was collected using comma EONs that has sensors similar to those of any modern smartphone including a road-facing camera, phone GPS, thermometers and 9-axis IMU. Additionally, the EON captures raw GNSS measurements and all CAN data sent by the car."

To use your data for training, you currently need to collect it with a [Comma 2 device](https://comma.ai/shop/products/two) (no support for version 3 yet). In the future, when true ground truth creation is implemented, you *might* be able to use a different device. Still, you'll need to adjust some hardware-related code (camera intrinsics, GNSS configuration in laika post-processing, etc). If you need more data than you can store on the device or Comma cloud or want to do it at scale, you can use a custom cloud server that re-implements Comma's API, called [retropilot-server](https://github.com/florianbrede-ayet/retropilot-server).


## Training pipeline

### Training loop


* **General:** 
  * Recurrent state of the GRU and desire are intialized with zeros. 
  * Traffic convention is hard-coded for left-hand-side driving.
  * A batch consists of `batch_size` sequences of length `seq_len` each from a different one-minute driving segment to reduce data correlation.

* **GRU training Logic:**
  * The first batch of each segment is not used to update the weights (recurrent state warmup).
  * The hidden state is preserved between batches of sequences within the same segment.
  * The hidden state is reset to zero when the current `batch_size` segments end.

<!-- TODO: continue from here-->

* **Visualization of predictions:**
  * Model predictions such as lanelines, road edges and path plans are visualized and logged to wandb after a certain interval of iterations, followed by the necessary validation of the trained model. 
  * To validate the predictions qualitatively we have also visualized the groundtruth.
  * Same Segments are used to visualize the model predictions and groundtruth. 
  * One segment from training set and one from validation set is used for visualization.

* **Wandb**
  * All the hyperparams, loss metrics and time taken by different modules of the training are logged and visualized in wandb.



### Data loading

In our preliminary experiments, we found that having more driving segments per batch is crucial for training. Batch size 8 (8 different segments per batch) leads to overfitting, while batch size 28 (maximum we could fit on our machine) gives a good performance.

PyTorch supports parallel batch loading *but not parallel sample loading*, so we implemented a custom data loader where each worker loads a single segment at a time, and a separate background process combines the results into a single batch. This is paired with pre-fetching and a (super hacky) synchronization mechanism to ensure the collation process doesn't block the shared memory until the main process has received the batch. 

Altogether this results in relatively low latency: ~150ms waiting + ~175ms transfer to GPU on our machine. Inter-process messaging instead of the hacky sync mechanism might bring the waiting down to <10ms. Speeding up transfer to GPU might be done through memory pinning, but it didn't work when I tried pinning tensors before pushing them to the shared memory queue. It probably has to be done on the consumer process side, but I am not sure how to keep it from slowing down the rest of the pipeline.

**NOTE:** The data loader requires *two CPU-cores (one train, one validation) per unit of batch size*, plus additional two CPU-cores for the main and background (collation) processes. Per-unit-of-batch-size cost could be brought down to 1 CPU-core if we implement stopping/restarting the train/validation workers as needed.

## How to Use

### System Requirements

- Ubuntu 20.04 LTS **with sudo** for compiling openpilot & ground truth creation. Training can probably be done on any Linux machine where PyTorch is supported.
- 50+ CPU cores, but more (~128-256) would mean better GPU utilization.
- GPU with at least 6 GB of memory.

### Installations
1. [Install openpilot](https://github.com/commaai/openpilot/tree/master/tools)
2. Clone the repo
```bash
git clone https://github.com/nikebless/openpilot-pipeline.git
```

3. Install conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
4. Install the repo's conda environment: <!-- TODO: update environment with imageio and moviepy -->

```bash
cd openpilot-pipeline/
conda env create -f environment.yml
```

### Run

1. Get the dataset in the [comma2k19](https://github.com/commaai/comma2k19) format available in a local folder. Either from comma2k19, or from your own collected data, as explained in the [data pipeline](#data-pipeline).
2. Run ground truth creation using [gt_hacky](gt_hacky) <!-- TODO: merge calibration extraction with gt_hacky -->
3. Set up [wandb](https://docs.wandb.ai/quickstart)
4. Run Training
* via slurm script
  ```bash
  sbatch train.sh --date_it <iteration_name> --recordings_basedir <dataset_dir>
  ```

* via slurm script
  ```bash
  python train.py --date_it <iteration_name> --recordings_basedir <dataset_dir>
  ```
The only required parameter are `--date_it` and `--recordings_basedir`, by running the above commands the default params will be used. If in case you want to alter the params. Detailed description of the parameters:

* `--batch_size` - batch size is equal to the number of workers used by dataloader. It should be decided depending on the number of cores of cpu available at the time of training. Currently we have tested the dataloder until batch size `28`. 
* `--date_it` - name of the training iteration.
* `--epochs` - number of epochs for training, default is 15
* `--grad_clip` - enable gradient clipping, default is infinity.
* `--l2_lambda` - weight decay value used in the adam optimizer, default is 1e-4.
* `--log_frequency` - after how many iterations you want to log the train loss to wandb and show it in the output, default is 100.
* `--lr` - learning rate, deafult is 0.001
* `--lrs_factor` - factor by which the learning rate is reduced by the scheduler, deafult is 0.75 
* `--lrs_min` - a lower bound on the learning rate for scheduler, default is 1e-6
* `--lrs_patience` - number of epochs with no improvement when learning rate is reduced by scheduler, default is 3
* `--lrs_thresh` - threshold for measuring new optimum by scheduler, default is 1e-4
* `--mhp_loss` - enable the multi hypothesis loss for training paths, by default distillation loss is enabled.
* `--no_recurr_warmup` - to enable the recurrent warmup False, by default it is True. 
* `--no_wandb` - disable the wandb logging, by default it is always on.
* `--recordings_basedir` - recordings or dataset path, default is our reccordings path.
* `--seed` - for the model reproducibility, default is 42
* `--seq_len` - length of sequence fetched by the dataloader for a batch, default is 100.
* `--split` - training and validation dataset split, default is 0.94
* `--val_frequency` - after how many iterations you want to enable the visulization of predictions by the trained model and validation.
### Using the model


0. Convert the model to ONNX format
```bash
cd train
python torch_to_onnx.py <model_path>
```
1. In simulation (Carla)
* Make sure that you have installed [openpilot](https://github.com/commaai/openpilot/tree/master/tools). 
* Make sure that cuda drivers are installed properly (`nvidia-smi` is available) 
* Go to the sim folder in cloned openpilot repo. 
  ```bash
  cd openpilot/tools/sim 
  ```
* Add your model in `openpilot/models` and rename it to `supercombo.onnx`.
* After that you need to edit the script [start_openpilot_docker.sh](https://github.com/commaai/openpilot/blob/master/tools/sim/start_openpilot_docker.sh), and add a bind mount at the end of the `docker run` command specifying the source and target path for the docker container.
```
Disclaimer: May be the current version of openpilot Carla is broken. And the bash script always pull the latest version of sim. So you can go back in the history (https://github.com/commaai/openpilot/pkgs/container/openpilot-sim)  and based on timeline of working commits change the tag in the docker run command in the script.
```
* Open two separate terminals and execute these bash scripts. 
  ```bash
  cd openpilot/tools/sim 

  ./start_carla.sh
  
  ./start_openpilot_docker.sh
  ```
2. In your car (via the Comma 2 device) — [Convert to DLC](doc/ONNX_to_DLC.md), where as comma 3 supports onnx.

## Our Results

* Visualization of the results via model trained with MHP (Multi-hypothesis loss) and KL divergence (Distillation). 
<table>
  <tr>
    <td>Likelihood model (~1h of training) </td>
     <td>Distillation model (~20h of training)</td>
  </tr>
  <tr>
    <td><img src="doc/MHP_vis.png" ></td>
    <td><img src="doc/distill_viz.png" ></td>
 </table>

## Technical improvement ToDos


**Important**
- [ ] Use drive calibration info in inputs transformation & for visualization
- [ ] Do not crash training when a single video failed to read

**Nice to haves**
- [ ] Better synchronization mechanism to speed up data loader


[^1]: Top 1 Overall Ratings, [2020 Consumer Reports](https://data.consumerreports.org/wp-content/uploads/2020/11/consumer-reports-active-driving-assistance-systems-november-16-2020.pdf)


