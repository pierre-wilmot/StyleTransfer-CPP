#include <iostream>
#include <torch/torch.h>

#include "args.h"
#include "MultiscaleStyleTransfer.h"
#include "ImageLoader.h"

int main(int ac, char **av)
{
  std::cout << "StyleTransfer++" << std::endl;

  // Define the accepted argument
  args::ArgumentParser parser("This is a test program.", "This goes after the options.");
  parser.SetArgumentSeparations(false, false, true, true);
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<std::string> contentArgument(parser, "content", "Path to the content image", {"content"});
  args::ValueFlag<std::string> styleArgument(parser, "style", "Path to the style image", {"style"}, args::Options::Required);
  args::ValueFlag<std::string> outputArgument(parser, "output", "Path to the output image", {"output"});
  args::ValueFlag<int> scalesArgument(parser, "scales", "Number of scales to use", {"scales"});

  // Parse command line arguments
  try
  {
    parser.ParseCLI(ac, av);
  }
  catch (const args::Help&)
  {
    std::cout << parser;
    return 0;
  }
  catch (args::Error e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Using device " << (torch::cuda::is_available() ? "CUDA" : "CPU") << std::endl;

  unsigned int scales(4);
  if (args::get(scalesArgument))
    scales = args::get(scalesArgument);
  std::cout << "Using " << scales << " scales" << std::endl;

  MultiscaleStyleTransfer model(scales);
  torch::load(model, "VGG.pt");
  std::cout << model << std::endl;
  model->to(device);

  torch::Tensor content;
  if (!args::get(contentArgument).empty())
  {
    content = imageToTensor(args::get(contentArgument));
    content = resizePreprocessedImage(content, 512, 512);
    content = content.to(device);
    tensorToImage(content, "content.png");
  }

  torch::Tensor style = imageToTensor(args::get(styleArgument));
  style = style.to(device);
  tensorToImage(style, "style.png");

  torch::Tensor canvas = torch::rand({3, 32, 32});
  canvas = canvas.to(device);
  for (float ratio : {8.0, 4.0, 2.0, 1.0})
  {
    canvas = resizePreprocessedImage(canvas, canvas.sizes()[1] * 2, canvas.sizes()[2] * 2);
    {
      if (!args::get(contentArgument).empty())
      {
	torch::Tensor scaledContent = resizePreprocessedImage(content, canvas.sizes()[1] , canvas.sizes()[2]);
	model->setContent(scaledContent);
      }
      torch::Tensor scaledStyle = resizePreprocessedImage(style, style.sizes()[1] / ratio , style.sizes()[2] / ratio);
      model->setStyle(scaledStyle);
    }
    model->optimise(canvas);

    std::string output = args::get(outputArgument).empty() ? "result.png" : args::get(outputArgument);
    tensorToImage(canvas, output);
  }

  return 0;
}
