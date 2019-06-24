#include <torch/extension.h>
#include "../../include/ImageLoader.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("imageToTensor", &imageToTensor, "Load an image into a Tensor using STB_IMAGE backend");
}
