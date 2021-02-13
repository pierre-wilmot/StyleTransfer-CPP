#include <iostream>
#include <torch/torch.h>

#include "StyleTransfer.h"
#include "ImageLoader.h"

int main(int ac, char **av)
{
  std::cout << "StyleTransfer++" << std::endl;

  if (ac < 3)
    {
      std::cout << "Usage: " << av[0] << " CONTENT_IMAGE STYLE_IMAGE1 [STYLE_IMAGE2 ...]" << std::endl;
      return 0;
    }

  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Using device " << (torch::cuda::is_available() ? "CUDA" : "CPU") << std::endl;
  StyleTransfer model;
  torch::load(model, "VGG.pt");
  std::cout << model << std::endl;
  model->to(device);

  torch::Tensor content = imageToTensor(av[1]);
  content = resizePreprocessedImage(content, 512, 512);
  content = content.to(device);
  tensorToImage(content, "content.png");
  for (int a(2) ; a < ac ; ++a)
    {
      torch::Tensor style = imageToTensor(av[a]);
      style = style.to(device);
      tensorToImage(style, "style.png");
      torch::Tensor canvas = torch::rand({3, 32, 32});
      canvas = canvas.to(device);
      for (float ratio : {8.0, 4.0, 2.0, 1.0})
	{
	  canvas = resizePreprocessedImage(canvas, canvas.sizes()[1] * 2, canvas.sizes()[2] * 2);
	  {
	    torch::Tensor scaledContent = resizePreprocessedImage(content, canvas.sizes()[1] , canvas.sizes()[2]);
	    model->setContent(scaledContent);
	    torch::Tensor scaledStyle = resizePreprocessedImage(style, style.sizes()[1] / ratio , style.sizes()[2] / ratio);
	    model->setStyle(scaledStyle);
	  }
    	  model->optimise(canvas);
	  tensorToImage(canvas, "result_" + std::to_string(a) + "_" + std::to_string(int(ratio)) + ".png");
	}
    }
  return 0;
}
