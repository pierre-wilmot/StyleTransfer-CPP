#include <iostream>
#include <torch/torch.h>

#include "StyleTransfer.h"

int main(int ac, char **av)
{
  std::cout << "StyleTransfer++" << std::endl;

  StyleTransfer model;
  std::cout << model << std::endl;

  torch::Tensor t = torch::rand({1, 3, 512, 512});
  torch::Tensor res = model->forward(t);
  std::cout << res.sizes() << std::endl;

  model->setStyle(t);

  t = torch::rand({1, 3, 512, 512});
  model->optimise(t);
  
  return 0;
}
