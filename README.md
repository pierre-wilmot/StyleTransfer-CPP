# StyleTransfer-CPP
[![Build Status](https://travis-ci.org/pierre-wilmot/StyleTransfer-CPP.svg?branch=master)](https://travis-ci.org/pierre-wilmot/StyleTransfer-CPP)

## Compiling
In order to compile you need to download the Libtorch library. Just run the script provided:
```sh download_torch.sh```
The rest of the process is classic CMake workflow:
```mkdir build ; cd build ; cmake .. && make```

## Running
In order to get anything running, we need to download the trained VGG weights and convert then to a format that can be loaded in C++.
To do that we use python and torchvision to get the model, and convert it to a C++ torch saved object.
To do so used the provided script:
```
pip install torchvision
cd tools/modelExporter
python3 python2CPP.py
```
This script will create a VGG.pt file that contains the trained VGG weight in a C++ format.
You need to copy or link this file in the folder from where you run your executable.
```
cd build
ln -s ../tools/modelExporter/VGG.pt .
```

Now everything is ready, you can run the StyleTransferCLI executable
```StyleTransferCLI PATH_TO_TOUR_IMAGE.png```