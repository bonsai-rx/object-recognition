# Bonsai - GenericNets
![logo](Resources/logo.svg)

## How to install

Bonsai.TensorFlow.ObjectRecognition can be downloaded through the Bonsai package manager. However, in order to use it for either CPU or GPU inference, you need to pair it with a compiled native TensorFlow binary. You can find precompiled binaries for Windows 64-bit at https://www.tensorflow.org/install/lang_c.

To use GPU TensorFlow (highly recommended for live inference), you also need to install the `CUDA Toolkit` and the `cuDNN libraries`. The current package was developed and tested with [CUDA v11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) and [cuDNN 8.2](https://developer.nvidia.com/cudnn). Additionally, make sure you have a CUDA [compatible GPU](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#support-hardware) with the latest NVIDIA drivers.

After downloading the native TensorFlow binary and cuDNN, you can follow these steps to get the required native files into the `Extensions` folder of your local Bonsai install:

1. The easiest way to find your Bonsai install folder is to right-click on the Bonsai shortcut > Properties. The path to the folder will be shown in the "Start in" textbox;
2. Copy `tensorflow.dll` file from either the CPU or GPU [tensorflow release](https://www.tensorflow.org/install/lang_c#download_and_extract) to the `Extensions` folder;
3. If you are using TensorFlow GPU, make sure to add the `cuda/bin` folder of your cuDNN download to the `PATH` environment variable, or copy all DLL files to the `Extensions` folder.

## How to use

This library is made to run inference on the pre-trained network `ssd_inception_v2_coco_2017_11_17`. You can [download this network from `tfhub`](https://tfhub.dev/nvidia/unet/industrial/class_10/1) using the `[tensorflow_hub](https://www.tensorflow.org/hub/installation)` python package. 
Inference on single frames can be run using the `PredictObject` operator. This node requires a path for a valid `.pb` file and a `.csv` file with human readable labels of objects identified by the network. 
