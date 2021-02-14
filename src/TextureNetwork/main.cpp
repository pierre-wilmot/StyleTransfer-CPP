#include <iostream>
#include <torch/torch.h>

#include "StyleTransfer.h"
#include "ImageLoader.h"

constexpr unsigned int SAMPLES = 10000;

class GeneratorImpl : public torch::nn::Module
{
public:
  GeneratorImpl(unsigned int nc = 64)
  {
    _c1 = register_module("c1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, nc, 3).padding(1)));
    _c2 = register_module("c2", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1)));
    _c3 = register_module("c3", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1)));
    _c4 = register_module("c4", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1)));
    _c5 = register_module("c5", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1)));
    _c6 = register_module("c6", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1)));
    _c7 = register_module("c7", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, 3, 3).padding(1)));
  }

  torch::Tensor forward(torch::Tensor const &x_)
  {
    torch::Tensor x = x_;
    x = torch::relu(_c1(x));
    x = torch::relu(_c2(x));
    x = torch::relu(_c3(x));
    x = torch::relu(_c4(x));
    x = torch::relu(_c5(x));
    x = torch::relu(_c6(x));
    x = _c7(x);
    return x;
  }

private:
  torch::nn::Conv2d _c1 = nullptr;
  torch::nn::Conv2d _c2 = nullptr;
  torch::nn::Conv2d _c3 = nullptr;
  torch::nn::Conv2d _c4 = nullptr;
  torch::nn::Conv2d _c5 = nullptr;
  torch::nn::Conv2d _c6 = nullptr;
  torch::nn::Conv2d _c7 = nullptr;
};
TORCH_MODULE(Generator);

int main(int ac, char **av)
{
  std::cout << "Texture Networks" << std::endl;
  if (ac != 2)
  {
    std::cout << "Usage: " << av[0] << " STYLE_IMAGE" << std::endl;
    return 0;
  }

  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Using device " << (torch::cuda::is_available() ? "CUDA" : "CPU") << std::endl;

  // Load style image
  torch::Tensor style = imageToTensor(av[1]);
  style = style.to(device);
  tensorToImage(style, "style.png");

  // Instanciate generator
  Generator generator;
  generator->to(device);
  std::cout << generator << std::endl;

  // Instanciate criterion
  StyleTransfer criterion;
  torch::load(criterion, "VGG.pt");
  criterion->to(device);
  std::cout << criterion << std::endl;

  // Set style for criterion
  criterion->setStyle(style);

  // Instanciate optimizer
  torch::optim::Adam optimizer(generator->parameters(), 1e-4);

  torch::Tensor noise = torch::rand({1, 3, 256, 256});
  noise = noise.to(device);
  std::cout << "noise.sizes() " << noise.sizes() << std::endl;
  for (int i(0) ; i < SAMPLES ; ++i)
  {
    optimizer.zero_grad();
    torch::Tensor stylised = generator(noise);
    if (i % 100 == 0)
      tensorToImage(stylised[0], std::to_string(i) + ".png");
    torch::Tensor loss = criterion->computeLoss(stylised);
    std::cout << loss.item<float>() << std::endl;
    loss.backward();
    optimizer.step();
  }

  return 0;
}
