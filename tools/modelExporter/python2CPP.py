# At the time of writting this code, I couldn't find a way to load the trained model directly in C++
# This piece of code creates a bridge between torchvision.models.vgg19(pretrained=True) and torch::load(Module)
# This is quite hacky and inneficient, but untill the loading is implemented in Pytoch C++, that will do ;)
# If you can think of a better solution, please let me know
# -- Pierre, 22nd June 2019

import torch
import torchvision
from torch.utils.cpp_extension import load

cpp = torch.utils.cpp_extension.load(name="print_cpp", sources=["python2CPP.cpp"])
print("Converting Python model to C++")
vgg = torchvision.models.vgg19(pretrained=True).features

cpp.set("conv1_1", vgg[0].weight, vgg[0].bias)
cpp.set("conv1_2", vgg[2].weight, vgg[2].bias)
cpp.set("conv2_1", vgg[5].weight, vgg[5].bias)
cpp.set("conv2_2", vgg[7].weight, vgg[7].bias)
cpp.set("conv3_1", vgg[10].weight, vgg[10].bias)
cpp.set("conv3_2", vgg[12].weight, vgg[12].bias)
cpp.set("conv3_3", vgg[14].weight, vgg[14].bias)
cpp.set("conv3_4", vgg[16].weight, vgg[16].bias)
cpp.set("conv4_1", vgg[19].weight, vgg[19].bias)
cpp.set("conv4_2", vgg[21].weight, vgg[21].bias)
cpp.set("conv4_3", vgg[23].weight, vgg[23].bias)
cpp.set("conv4_4", vgg[25].weight, vgg[25].bias)
cpp.set("conv5_1", vgg[28].weight, vgg[28].bias)
