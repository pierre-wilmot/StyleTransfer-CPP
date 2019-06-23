#include <torch/extension.h>
#include "../../include/StyleTransfer.h"

void set(std::string const &name, torch::Tensor const &weights, torch::Tensor const &bias)
{
  StyleTransfer m;
  try
    {
      // Load current model
      torch::load(m, "VGG.pt");
    }
  catch (c10::Error const &e)
    {
      // Pass if there's no saveed state yet
    }

  for (auto &module : m->named_modules())
    {
      if (module.key() == name)
	{
	  torch::nn::Conv2d c = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(module.value());
	  c->weight.copy_(weights);
	  c->bias.copy_(bias);
	  torch::save(m, "VGG.pt");
	  return;
	}
    }
  std::cerr << "Could not find [" << name << "] in the model" << std::endl;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("set", &set, "Load weights and bias in a given convolution layer");
}
