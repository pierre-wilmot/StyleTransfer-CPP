#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize.h"

#include <torch/torch.h>


torch::Tensor &preprocess(torch::Tensor &t)
{
  // Imagenet Preprocessing
  t[0].sub_(0.485).div_(0.229);
  t[1].sub_(0.456).div_(0.224);
  t[2].sub_(0.406).div_(0.225);
  return t;
}

torch::Tensor &deprocess(torch::Tensor &t)
{
  // Imagenet deprocessing
  t[0].sub_(-0.485/0.229).div_(1/0.229);
  t[1].sub_(-0.456/0.224).div_(1/0.224);
  t[2].sub_(-0.406/0.225).div_(1/0.225);
  return t;
}

torch::Tensor imageToTensor(std::string const &path)
{
  int x, y, n;
  unsigned char *data = stbi_load(path.c_str(), &x, &y, &n, 3);
  if (!data)
    throw std::runtime_error("Failled to load [" + path + "] -- " + stbi_failure_reason());
  torch::Tensor t = torch::zeros({y, x, 3}).to(caffe2::TypeMeta::Make<unsigned char>());;
  memcpy(t.data_ptr(), data, x * y * 3);
  stbi_image_free(data);
  t = t.transpose(0, 2);
  t = t.transpose(1, 2);
  t = t.to(caffe2::TypeMeta::Make<float>());;
  t.div_(256.0f);
  return preprocess(t);
}

int tensorToImage(torch::Tensor const &image, std::string const &path)
{
  assert(image.dim() == 3);
  torch::Tensor t = image.clone();
  deprocess(t);
  t = t.transpose(2, 1);
  t = t.transpose(2, 0);
  t.clamp_(0, 1);
  t.mul_(255);
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
  if (!t.is_contiguous())
    t = t.clone();
  assert(t.is_contiguous());
  torch::Tensor output = torch::zeros({h, w, 3});
  auto const &s = t.sizes();
  std::cout << s << std::endl;
  stbir_resize_float(t.data<float>(), s[1] , s[0] ,  s[1] * s[2] * sizeof(float),
		     output.data<float>(), w, h, w * 3 * sizeof(float),
		     3);
  output = output.transpose(0, 2);
  output = output.transpose(1, 2);
  return output;
}

torch::Tensor resizePreprocessedImage(torch::Tensor const &image, unsigned int w, unsigned int h)
{
  torch::Tensor t = image.clone();
  deprocess(t);
  t = resizeImage(t, w, h);
  return preprocess(t);
}
