#include <iostream>
#include <torch/torch.h>

#include "StyleTransfer.h"
#include "ImageLoader.h"

int main(int ac, char **av)
{
  std::cout << "StyleTransfer++" << std::endl;

  if (ac != 2)
    {
      std::cout << "Usage: " << av[0] << " STYLE_IMAGE" << std::endl;
      return 0;
    }
  
  StyleTransfer model;
  std::cout << model << std::endl;

  torch::Tensor style = imageToTensor(av[1]).unsqueeze(0);
  model->setStyle(style);
  
  torch::Tensor canvas = torch::rand({1, 3, 512, 512});
  model->optimise(canvas);
  tensorToImage(canvas, "result.png");
  
  return 0;
}
