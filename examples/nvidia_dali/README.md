# DALI Optimization integration with Torchserve models


The NVIDIA Data Loading Library (DALI) is a library for data loading and pre-processing to accelerate deep learning applications. It provides a collection of highly optimized building blocks for loading and processing image, video and audio data.

Here, we serve torchserve models with DALI pipeline for optimizing the inference

Refer to [NVIDIA-DALI-Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) for detailed information


### Installation :

```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
```

### Define and Build DALI Pipeline

In DALI, any data processing task has a central object called Pipeline. 

Navigate to `cd ./serve/examples/nvidia_dali`. Run the python file which serializes the Dali Pipeline and saves the file as `./serve/examples/nvidia_dali/model_repository/model.dali`


```bash
python serialiize_dali_pipeline.py	
```

Refer to [NVIDIA-DALI](https://github.com/NVIDIA/DALI) for more details on DALI pipeline.

### Sample commands to create a resnet-18 eager mode model archive with dali pipeline and dali_config file, register it on TorchServe and run image prediction


Navigate to `cd serve` directory and run the below commands

	
To download the resnet .pth file

```bash
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

The following command will create a .mar extension file where we also add the model.dali file and dali_config.json file in it.

```bash
torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./examples/image_classifier/resnet_18/model.py --serialized-file resnet18-f37072fd.pth --handler image_classifier --extra-files ./examples/image_classifier/index_to_name.json,./examples/nvidia_dali/model_repository/model.dali,./examples/nvidia_dali/dali_config.json
```


Create a new directory `model_store` and move the model-archive file

```bash
mkdir model_store
mv resnet-18.mar model_store/
```


Run the following command in your terminal to set the environment variable for DALI_PREPROCESSING

```bash
export DALI_PREPROCESSING=true
```


Start the torchserve
```bash
torchserve --start --model-store model_store --models resnet-18=resnet-18.mar
```


Get the inference for a sample image using the below command 
```bash
curl http://127.0.0.1:8080/predictions/resnet-18 -T ./examples/image_classifier/kitten.jpg
```



