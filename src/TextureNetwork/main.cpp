#include <iostream>
#include <torch/torch.h>

#include "StyleTransfer.h"
#include "ImageLoader.h"

constexpr unsigned int SAMPLES = 5000;

class ResidualBlockImpl : public torch::nn::Module
{
public:
  ResidualBlockImpl(unsigned int nc = 64)
  {
    _c1 = register_module("c1", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).padding_mode(torch::kCircular)));
    _c2 = register_module("c2", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).padding_mode(torch::kCircular)));
    _c3 = register_module("c3", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).padding_mode(torch::kCircular)));
    _c4 = register_module("c4", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).padding_mode(torch::kCircular)));
    _c5 = register_module("c5", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).padding_mode(torch::kCircular)));
    _c6 = register_module("c6", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).padding_mode(torch::kCircular)));
    _c7 = register_module("c7", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).padding_mode(torch::kCircular)));
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
    return x + x_;
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
TORCH_MODULE(ResidualBlock);

class GeneratorImpl : public torch::nn::Module
{
public:
  GeneratorImpl(unsigned int nc = 64)
  {
    _c1 = register_module("c1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, nc, 3).padding(1).padding_mode(torch::kCircular)));
    _c2 = register_module("c2", ResidualBlock(64));
    _c3 = register_module("c3", ResidualBlock(64));
    _c4 = register_module("c4", ResidualBlock(64));
    _c5 = register_module("c5", ResidualBlock(64));
    _c6 = register_module("c6", ResidualBlock(64));
    _c7 = register_module("c7", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, 3, 1).padding(0)));
  }

  torch::Tensor forwardUntillStep(torch::Tensor const &x16, torch::Tensor const &x32, torch::Tensor const &x64,
				  torch::Tensor const &x128, torch::Tensor const &x256, torch::Tensor const &x512)
  {
    torch::nn::functional::InterpolateFuncOptions options;
    options.scale_factor(std::vector<double>({2, 2})).mode(torch::kNearest);
    torch::Tensor x = x16;
    x = torch::relu(_c1(x)); // 16
    if (_step == 1) return x;
    x = torch::nn::functional::interpolate(x, options);
    x += x32;
    x = torch::relu(_c2(x)); // 32
    if (_step == 2) return x;
    x = torch::nn::functional::interpolate(x, options);
    x += x64;
    x = torch::relu(_c3(x)); // 64
    if (_step == 3) return x;
    x = torch::nn::functional::interpolate(x, options);
    x += x128;
    x = torch::relu(_c4(x)); // 128
    if (_step == 4) return x;
    x = torch::nn::functional::interpolate(x, options);
    x += x256;
    x = torch::relu(_c5(x)); // 256
    if (_step == 5) return x;
    x = torch::nn::functional::interpolate(x, options);
    x += x512;
    x = torch::relu(_c6(x)); // 512
    return x;
  }

  torch::Tensor forward(torch::Tensor const &x16, torch::Tensor const &x32, torch::Tensor const &x64,
			torch::Tensor const &x128, torch::Tensor const &x256, torch::Tensor const &x512)
  {
    torch::Tensor x = forwardUntillStep(x16, x32, x64, x128, x256, x512);
    x = _c7(x);
    return x;
  }

  void nextStep()
  {
    _step++;
  }

private:
  torch::nn::Conv2d _c1 = nullptr;
  ResidualBlock _c2 = nullptr;
  ResidualBlock _c3 = nullptr;
  ResidualBlock _c4 = nullptr;
  ResidualBlock _c5 = nullptr;
  ResidualBlock _c6 = nullptr;
  torch::nn::Conv2d _c7 = nullptr;
  unsigned int _step = 1;
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

  // Instanciate generator
  unsigned int nc = 64;
  Generator generator(nc);
  generator->to(device);
  std::cout << generator << std::endl;

  // Instanciate criterion
  StyleTransfer criterion;
  torch::load(criterion, "VGG.pt");
  criterion->to(device);
  std::cout << criterion << std::endl;

   // Instanciate optimizer
  torch::optim::Adam optimizer(generator->parameters(), 1e-4);


  for (int scale_ratio : {32, 16, 8, 4, 2, 1})
  {
    torch::Tensor scaled_style = resizePreprocessedImage(style, float(style.sizes()[2]) / scale_ratio, float(style.sizes()[1]) / scale_ratio);
    std::cout << "Scale style " << scaled_style.sizes() << std::endl;
    // Set style for criterion
    criterion->setStyle(scaled_style);
    tensorToImage(scaled_style, "style.png");

    for (int i(0) ; i < SAMPLES ; ++i)
    {
      torch::Tensor noise16 = torch::rand({1, 3, 16, 16}).to(device);
      torch::Tensor noise32 = torch::rand({1, nc, 32, 32}).to(device);
      torch::Tensor noise64 = torch::rand({1, nc, 64, 64}).to(device);
      torch::Tensor noise128 = torch::rand({1, nc, 128, 128}).to(device);
      torch::Tensor noise256 = torch::rand({1, nc, 256, 256}).to(device);
      torch::Tensor noise512 = torch::rand({1, nc, 512, 512}).to(device);

      optimizer.zero_grad();
      torch::Tensor stylised = generator(noise16, noise32, noise64, noise128, noise256, noise512);
      if (i % 100 == 0)
	tensorToImage(stylised[0], std::to_string(i) + ".png");
      torch::Tensor loss = criterion->computeLoss(stylised);
      std::cout << loss.item<float>() << std::endl;
      loss.backward();
      optimizer.step();
    }

    generator->nextStep();
    torch::save(generator, "generator.pt");
  }

  return 0;
}
