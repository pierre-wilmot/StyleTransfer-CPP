#include <iostream>
#include <torch/torch.h>

#include "StyleTransfer.h"

int main(int ac, char **av)
{
  std::cout << "StyleTransfer++" << std::endl;

  StyleTransfer model;
  std::cout << model << std::endl;
  
  return 0;
}
