#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize.h"

#include <torch/torch.h>

torch::Tensor imageToTensor(std::string const &path)
{
  int x, y, n;
  float *data = stbi_loadf(path.c_str(), &x, &y, &n, 3);
  if (!data)
    throw std::runtime_error("Failled to load [" + path + "] -- " + stbi_failure_reason());
  torch::Tensor t = torch::zeros({y, x, 3});
  memcpy(t.data_ptr(), data, x * y * 3 * sizeof(float));
  
  stbi_image_free(data);
  t = t.transpose(0, 2);
  t = t.transpose(1, 2);
  return t;
}

int tensorToImage(torch::Tensor const &image, std::string const &path)
{
  assert(image.dim() == 3);
  torch::Tensor t = image.transpose(2, 1);
  t = t.transpose(2, 0);
  t.mul_(256);
  t = t.to(caffe2::TypeMeta::Make<unsigned char>());
  assert(t.is_contiguous());  
  // [H, W, C]
  auto const &s = t.sizes();
  return stbi_write_png(path.c_str(), s[1], s[0], s[2], t.data_ptr(), s[1] * s[2]);
}

torch::Tensor resizeImage(torch::Tensor const &image, unsigned int w, unsigned int h)
{
  assert(image.dim() == 3);
  torch::Tensor t = image.transpose(2, 1);
  t = t.transpose(2, 0);
  assert(t.is_contiguous());
  torch::Tensor output = torch::zeros({h, w, 3});
  auto const &s = t.sizes();
  std::cout << s << std::endl;
  stbir_resize_float(t.data<float>(), s[1] , s[0] ,  s[1] * s[2] * sizeof(float),
		     output.data<float>(), w, h, w * 3 * sizeof(float),
		     3);
  output = output.transpose(0, 2);
  output = output.transpose(1, 2);
  output.clamp_(0, 1);
  return output;
}
