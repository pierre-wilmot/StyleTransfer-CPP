#include <iostream>
#include <thread>
#include <SDL.h>

#include "args.h"
#include "StyleTransfer.h"
#include "ImageLoader.h"


class StyleTransferGUI : public StyleTransferDelegate
{
public:
  StyleTransferGUI()
  {
    //Start SDL
    SDL_Init( SDL_INIT_EVERYTHING );
    _win = SDL_CreateWindow("Hello World!", 100, 100, 512, 512, SDL_WINDOW_SHOWN);
    _ren = SDL_CreateRenderer(_win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    _tex = SDL_CreateTexture(_ren, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, 512, 512);
    _running = true;
  }

  virtual ~StyleTransferGUI()
  {
    //Quit SDL
    SDL_DestroyTexture(_tex);
    SDL_DestroyRenderer(_ren);
    SDL_DestroyWindow(_win);
    SDL_Quit();
  }

  void onUpdate(torch::Tensor t) override
  {
    _t = t;
  }

  void onFinished(torch::Tensor t) override
  {
    _running = false;
  }

  void refresh()
  {
    if (_t.defined())
      {
	unsigned int tensor_width = _t.sizes()[2];
	unsigned int tensor_height = _t.sizes()[1];
	int texture_width(0);
	int texture_height(0);
	SDL_QueryTexture(_tex, nullptr, nullptr, &texture_width, &texture_height);
	if (texture_width != tensor_width || texture_height != tensor_height)
	{
	  SDL_DestroyTexture(_tex);
	  _tex = SDL_CreateTexture(_ren, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, tensor_width, tensor_height);
	}

	torch::Tensor t = exportPreprocessedToSDL(_t, tensor_width, tensor_height);
	int pitch;
	SDL_LockTexture(_tex, nullptr, &_pixels, &pitch);
	memcpy(_pixels, t.data_ptr(), tensor_width * tensor_height * 4);
	SDL_UnlockTexture(_tex);

	SDL_RenderClear(_ren);
	SDL_RenderCopy(_ren, _tex, NULL, NULL);
	SDL_RenderPresent(_ren);
      }
  }

  void run()
  {
    SDL_Event event;
    while (_running)
      {
	while (SDL_PollEvent(&event))
	  {
	    if (event.type == SDL_QUIT)
	      return;
	  }
	refresh();
	SDL_Delay(500);
      }
  }

  bool running() const
  {
    return _running;
  }

  void setRunning()
  {
    _running = true;
  }

private:
  SDL_Window *_win;
  SDL_Renderer *_ren;
  SDL_Texture *_tex;
  void *_pixels;
  bool _running;
  torch::Tensor _t;
};

int main(int ac, char **av)
{
  std::cout << "StyleTransfer++" << std::endl;
  // Define the accepted argument
  args::ArgumentParser parser("This is a test program.", "This goes after the options.");
  parser.SetArgumentSeparations(false, false, true, true);
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<std::string> contentArgument(parser, "content", "Path to the content image", {"content"});
  args::ValueFlag<std::string> styleArgument(parser, "style", "Path to the style image", {"style"}, args::Options::Required);

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

  StyleTransfer model;
  torch::load(model, "VGG.pt");
  model->to(device);
  std::cout << model << std::endl;

  StyleTransferGUI gui;
  model->setDelegate(&gui);

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
      gui.setRunning();
      std::thread t( [&]() { model->optimise(canvas); } );
      gui.run();
      model->stopOptmising();
      std::cout << model << std::endl;
      t.join();
    }

  tensorToImage(canvas, "result.png");
  torch::Tensor tilled = torch::cat(std::vector<torch::Tensor>({canvas, canvas}), 2);
  tilled = torch::cat(std::vector<torch::Tensor>({tilled, tilled}), 1);
  tensorToImage(tilled, "tilled.png");

  SDL_Delay(1000);
  return 0;
}
