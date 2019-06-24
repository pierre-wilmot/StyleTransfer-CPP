import sys
import torch
from torch.utils.cpp_extension import load
import visdom

cpp = torch.utils.cpp_extension.load(name="image_cpp", sources=["CPPImageLoader.cpp"])
print("Loading images using C++")

if len(sys.argv) != 2:
    print("Usage : " + sys.argv[0] + " IMAGE_FILE")
    sys.exit(1)
    
image = cpp.imageToTensor(sys.argv[1])
viz = visdom.Visdom()
viz.image(image)
