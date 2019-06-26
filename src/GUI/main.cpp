#include <iostream>
#include <SDL.h>
#include "ImageLoader.h"

int main(int ac, char **av)
{
  if (ac != 2)
    {
      std::cerr << "Usage: " << av[0] << " IMAGE_TO_DISPLAY" << std::endl;
      return 0;
    }
  
  //Start SDL
  SDL_Init( SDL_INIT_EVERYTHING );

  SDL_Window *win = SDL_CreateWindow("Hello World!", 100, 100, 512, 512, SDL_WINDOW_SHOWN);
  if (win == nullptr)
    {
      std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
      SDL_Quit();
      return 1;
    }

  SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (ren == nullptr)
    {
      SDL_DestroyWindow(win);
      std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
      SDL_Quit();
      return 1;
    }
  
  torch::Tensor t = exportPreprocessedToSDL(imageToTensor(av[1]), 512, 512);
  auto tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, 512, 512);
  Uint32* pixels = nullptr;
  int pitch, w, h = 0;
  unsigned int format;  
  SDL_QueryTexture(tex, &format, nullptr, &w, &h);
  SDL_LockTexture(tex, nullptr, (void**)&pixels, &pitch);
  memcpy(pixels, t.data_ptr(), 512 * 512 * 4);
  SDL_UnlockTexture(tex);

  SDL_RenderClear(ren);
  SDL_RenderCopy(ren, tex, NULL, NULL);
  SDL_RenderPresent(ren);
  SDL_Delay(1000);
      
  //Quit SDL
  SDL_DestroyRenderer(ren);
  SDL_DestroyWindow(win);
  SDL_Quit();
    
  return 0;
}
