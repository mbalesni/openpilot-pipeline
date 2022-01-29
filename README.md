# openpilot-pipeline

Comma AI's Advanced Driver-Assitance System (ADAS) [Openpilot](https://github.com/commaai/openpilot) is open-source, but the code to create training data and train the models is not. We attempt to re-create & open-source their full pipeline.

## About @nikebless
- where this project is right now? (distillation only, no true training)

## Model
The neural network architecurre of the model consists of a Convolutional feature extractor which is a Resnet based custom feature extractor, followed by a GRU (used to capture the temporal context) and at the end we have different outputs heads which comprises of fully connnected layers responisble for outputs like paths, lanelines road edges and other meta information which is explained in detail below. 

<table>
  <tr>
    <td>CNN Feature Extractor</td>
     <td>GRU</td>
     <td>Output Heads</td>
  </tr>
  <tr>
    <td><img src="images/Conv_extractor.png" width=100 height=250></td>
    <td><img src="images/GRU.png" width=300 height=250></td>
    <td><img src="images/output_heads.png"  height=250></td>
  </tr>
 </table>

The model visualized above is taken from [Openpilot 0.8.10 Release](https://github.com/commaai/openpilot/tree/v0.8.10/models) and you can find the same 'supercombo.onnx' model in this [repository ](https://github.com/nikebless/openpilot-pipeline/tree/main/common/models). 

The visualization tool used for onnx architecures: [netron](https://github.com/lutzroeder/netron) and the installation instructions can be found under the parent repository.

Detailed definations of the inputs and outputs of the model are mentioned [here](https://github.com/commaai/openpilot/tree/master/models). 
### Inputs
As mentioned above we can divide the model architecture into two parts encoder and decoder. Where the encoder part is made up of CNN feature extractor and a GRU, whereas decoder is mutliheaded fully connected layers responsible for outputs. 

The primary input to the model are two consecutive frames which are converted into YUV 4:2:0 format and stacked to form a tensor of shape `(1,12,128,256)`.
* ##### YUV 4:2:0
  * It is a color subsampling technique.
  * Ecodes only 20% of the color infromation.
  * Saves memory footprint
  * Computationally efficient. 

Apart from the images there are three more inputs which are fed to the model in the intermediate stage where extracted features are passed into the GRU. So the GRU takes in extracted CNN features, desire, traffic convention and recurrent state. 
* ##### Desire
  * Desire in general for one sample is a 1-D vector with shape of `(1,8)`. In simpler terms it can be interpreted by the desirable actions performed by the driver during a drive scenario when openpilot is engaged. 
![desire](images/desire.png?raw=true)
More concrete infromation can be obtained from [here](https://github.com/commaai/cereal/blob/5c64eaab789dfa67a210deaa2530788843474999/log.capnp). 
* ##### Traffic Convention
  * As similar to desire for one sample it is a 1-D vector of shape `(1,2)` which is one hot encoded according to the LHD(Right Hand Driving) and RHD(Right Hand Driving). 
* ##### Recurrent State
  * This is basically taken from the output of the network which is refeed again to the the GRU. 

### Outputs
At the last part of the architecure all the ouputs from the fully connected branches are concatenated, thus the output of the model is of shape `(n,6472)` where n is the batch size.
The output vector is parsed according to index and the outputs can be obtained. The method to parse the output vector can be found in the older versions of the [openpilot](https://github.com/commaai/openpilot/blob/v0.8.5/selfdrive/modeld/models/driving.cc) and can be obtained [via](https://github.com/commaai/openpilot/tree/master/models). All the outputs are predicted in the form of mean, standard deviation and logits and regressed over time. All the predictions like path plans, lanelines, road_edges  are carried out for future 33 timestamps which spans upto 10 seconds from the current frame and each time stamp is associated with a distance from 0 to 192 meters. You can find the refference for that in the openpilot code with arguments named **T_IDXS** and **X_IDXS**. All the outputs from the model are predicted in comma's so called calibrated frame of refference, which is explained in detail [here](https://github.com/commaai/openpilot/tree/master/common/transformations).
* ##### Paths Plans
  * The model predicts 5 potential desired plans with the shape of $(n, 4955)$, which can be further splitted into 5 equal arrays with shape `(1,991)`. 
  * Further `(1,991)` is divided into paths and path probabilty.
  * Each path is associated with a probability.
  * To obtain the actual meaningful arrays for one path `(1,990)` can be reshaped to `(1,2,33,15)`.
  * `[:, 0, :, :]` = mean and  std = `[:, 1, :, :]`, for the following values.
  
* ##### Lanelines 
  * Modle predicts 4 potential lanelines, where the shape of the parsed vector is `(n,528)`.
  * This `(n,528)` can be further reshaped into `(4,2,33,2)`.
  * The four lanelines can be named as outer_left, inner_left, inner_right, outer_right.
  * The ego vehicle is always in the middle lane(inner_left and inner_right).
* ##### Laneline Probabilites 
  * Probability that each of the 4 lanes exist.
* ##### Road Edges
  * Model predicts two road edges left and right. 

* ##### Leads
* ##### Leads Probabilites 
* ##### Desire State

* ##### Meta

* ##### Pose

* ##### Recurrent State


### Technicalities: onnx2pytorch @nikebless

### Loss functions
- As mentioned above we are doing model distillation and currently training for just paths. All the predictions are in the terms of mean, standard deviation and logits. So we are first creating distribution [objects](https://pytorch.org/docs/1.7.1/distributions.html) and then calculating KL divergence for those distribution objects. 

- In the case when we are not using the model distillation we can use Gaussian or Laplacian NLL losses for all the regression outputs. For all the Multihypothesis strategies we can implement a basic-winner-to-take-all loss. 

- If we try to train for more than one task, the total loss can be calculated by summing all the losses for the tasks or task-loss balancing strategies can be implemeneted for refine results.  

## Data pipeline @nikebless
- gt_hacky — distilling existing model
- gt_real (not finished yet)

## Training pipeline

### Training loop


* **General:** 
  * A batch from the dataloader consists of stacked frames, groundtruth of plans and plan probabilities and a tensor which will define when a segment is finished. 
  * Recurrent state of the GRU and desire are intialized with zeros. 
  * Traffic convention is initialized as a one-hot vector with LHD.
  * An iteration is completed when a batch is processed.  
  * For a single iteration samples are processed by looping over sequence length and step loss is calculated and in the end batch loss is calculated by diving it by sequence lenght and batch size. 
  * By dividing the train loss and validation loss by sequence length and batch size, we can  compare loss metrics for mulitple runs with different sequence length and batch size.

* **GRU training Logic:**
  * Recurrent Wramup, is introduced when the training starts, basically gradients are not propagated back for certain iterations, where as the recurrent state is updated.
  * After achievening the recurrent warmup stage, the training is resumed normally.
  * Recurrent state is updated everytime while iterating over batches of sequence except for the last sequence and it is detached from the computational graph.
  * The final hidden state is used a initial hidden state for next iteration.
  * When a segment is finished while fetching data by the dataloader recurrent state is reset to zeros. 

* **Visualization of predictions:**
  * Model predictions such as lanelines, road edges and path plans are visualized and logged to wandb after a certain interval of iterations, followed by the necessary validation of the trained model. 
  * To validate the predictions qualitatively we have also visualized the groundtruth.
  * Same Segments are used to visualize the model predictions and groundtruth. 
  * One segment from training set and one from validation set is used for visualization.

* **Wandb**
  * All the hyperparams, loss metrics and time taken by different modules of the training are logged and visualized in wandb.



### Data loading @nikebless
- How we create batches (mention requirement of 1 CPU per 1 sample)

## How to Use

### System Requirements @nikebless
- for GT creation we need sudo + probably Ubuntu 20.04 (for compiling openpilot)
- for training, no sudo required, anything that can run pytorch

### Installations
1. [Install openpilot](https://github.com/commaai/openpilot/tree/master/tools) (for LogReader mainly)
2. Install conda environment.
*
  ```
  git clone https://github.com/nikebless/openpilot-pipeline.git
  ```
* Install conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

*
  ```
  cd openpilot-pipeline/
  conda env create -f environment.yml
  ```

### Running

1. Get dataset: @nikebless
  - simplest: comma2k19
  - custom data small scale: copy from comma device
  - custom data large scale: retropilot
2. Run ground truth creation @nikebless
  - gt_hacky + parse_logs (mention we're not using RPY yet)
3. Set up wandb @gauti
4. Run Training
* via slurm script
  ```
  sbatch train.sh --date_it <iteration_name> --recordings_basedir <dataset_dir>
  ```

* via slurm script
  ```
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
### Using the model @gauti


0. Save model in ONNX
1. In simulation (Carla)
2. In the Comma 2 device — [Convert to DLC](doc/ONNX_to_DLC.md), where as comma 3 supports onnx.

## Our Results @gauti

- (maybe) loss metrics
- visualized predictions

## Technical improvement ToDos

- [ ] Do not crash training when a single video failed to read
