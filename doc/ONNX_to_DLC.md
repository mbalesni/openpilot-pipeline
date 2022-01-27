# ONNX => DLC


## Official Instruction
https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/getting-started

I had a lot of problems when I tried to follow to the official instruction. 
Thus I outlined alternative instructions that works. 

## Prerequisites
* install Ubuntu [18.04](https://releases.ubuntu.com/18.04/ubuntu-18.04.6-desktop-amd64.iso)
* download [**android studio**](https://developer.android.com/studio/index.html)
	* extract archive 
	* launch **studio.sh** inside extracted files
	* install android SDK
	* install android NDK.
		1. In **welcome android screen** select **projects**
		2. More actions
		3. SDK manager
		4. SDK tools
		5. Select NDK and press **apply**

*
  ```
  sudo apt install git
  sudo apt install cmake
  sudo apt install python3-pip python3-dev libprotobuf-dev protobuf-compiler
  pip3 install --upgrade protobuf
  pip3 install packaging
  ```

## Install Qualcomm SDK
* Download the latest version of the [SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/tools)
* extract files
* 
  ```
  sudo apt-get install python3-dev python3-matplotlib python3-numpy python3-protobuf python3-scipy python3-skimage python3-sphinx wget zip
  chmod 777 ~/snpe-sdk/bin/dependencies.sh
  ~/snpe-sdk/bin/dependencies.sh
  ```
* change from **python** to **python3** here ~/snpe-sdk/bin/check_python_depends.sh

  ```
  source ~/snpe-sdk/bin/check_python_depends.sh
  ```
* Install onnx
  ```
  git clone --recursive https://github.com/onnx/onnx.git
  export ONNX_DIR=\<location of onnx repository\>
  export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
  pip3 install onnx
  ```
* Setup SNPE
  ```
  export SNPE_ROOT=~/snpe-sdk/
  source ~/snpe-sdk/bin/envsetup.sh -o $ONNX_DIR
  ```

## Usage
Here is the [instuction](https://developer.qualcomm.com/sites/default/files/docs/snpe/model_conv_onnx.html) how to use the convertor. 

