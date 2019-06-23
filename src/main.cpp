#include <iostream>
#include <torch/torch.h>

int main(int ac, char **av)
{
  std::cout << "StyleTransfer++" << std::endl;

  torch::Tensor t = torch::zeros({5, 5});
  std::cout << t << std::endl;
  
  return 0;
}
