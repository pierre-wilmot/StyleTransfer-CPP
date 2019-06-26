#include <iostream>
#include <thread>
#include <SDL.h>
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
    int w, h; unsigned int format;
    SDL_QueryTexture(_tex, &format, nullptr, &w, &h);
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
	torch::Tensor t = exportPreprocessedToSDL(_t, 512, 512);
	int pitch;
	SDL_LockTexture(_tex, nullptr, &_pixels, &pitch);
	memcpy(_pixels, t.data_ptr(), 512 * 512 * 4);
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
  if (ac != 3)
    {
      std::cout << "Usage: " << av[0] << " CONTENT_IMAGE STYLE_IMAGES" << std::endl;
      return 0;
    }
  
  StyleTransfer model;
  torch::load(model, "VGG.pt");
  std::cout << model << std::endl;

  StyleTransferGUI gui;
  model->setDelegate(&gui);

  torch::Tensor content = imageToTensor(av[1]);
  content = resizePreprocessedImage(content, 512, 512);
  torch::Tensor style = imageToTensor(av[2]);

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
      gui.setRunning();
      std::thread t( [&]() { model->optimise(canvas); } );
      gui.run();
      model->stopOptmising();
      t.join();
    }
  tensorToImage(canvas, "result.png");
  SDL_Delay(1000);

  return 0;
}
