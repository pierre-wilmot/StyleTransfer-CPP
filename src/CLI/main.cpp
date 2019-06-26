#include <iostream>
#include <torch/torch.h>

#include "StyleTransfer.h"
#include "ImageLoader.h"

int main(int ac, char **av)
{
  std::cout << "StyleTransfer++" << std::endl;

  if (ac < 2)
    {
      std::cout << "Usage: " << av[0] << " STYLE_IMAGES ..." << std::endl;
      return 0;
    }
  
  StyleTransfer model;
  torch::load(model, "VGG.pt");
  std::cout << model << std::endl;

  torch::Tensor content = imageToTensor("/home/wilmot_p/Pictures/avatar.jpg");
  content = resizePreprocessedImage(content, 512, 512);
  for (int a(1) ; a < ac ; ++a)
    {
      torch::Tensor style = imageToTensor(av[a]);
      torch::Tensor canvas = torch::rand({3, 32, 32});
      for (float ratio : {8.0, 4.0, 2.0, 1.0})
	{
	  canvas = resizePreprocessedImage(canvas, canvas.sizes()[1] * 2, canvas.sizes()[2] * 2);
	  {
	    torch::Tensor scaledContent = resizePreprocessedImage(content, canvas.sizes()[1] , canvas.sizes()[2]);
	    model->setContent(scaledContent);
	    torch::Tensor scaledStyle = resizePreprocessedImage(style, style.sizes()[1] / ratio , style.sizes()[2] /ratio);
	    model->setStyle(scaledStyle);
	  }
    	  model->optimise(canvas);
	  tensorToImage(canvas, "result_" + std::to_string(a) + "_" + std::to_string(int(ratio)) + ".png");
	}
      

    }
  return 0;
}
