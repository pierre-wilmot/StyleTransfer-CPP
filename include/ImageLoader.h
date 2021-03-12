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
  t = t.cpu();
  t = t.to(caffe2::TypeMeta::Make<unsigned char>());
  if (!t.is_contiguous())
    t = t.contiguous();
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
    t = t.contiguous();
  assert(t.is_contiguous());
  torch::Tensor output = torch::zeros({h, w, 3});
  auto const &s = t.sizes();
  stbir_resize_float(t.cpu().data_ptr<float>(), s[1] , s[0] ,  s[1] * s[2] * sizeof(float),
		     output.data_ptr<float>(), w, h, w * 3 * sizeof(float),
		     3);
  output = output.transpose(0, 2);
  output = output.transpose(1, 2);
  return output.to(image.device());
}

torch::Tensor resizePreprocessedImage(torch::Tensor const &image, unsigned int w, unsigned int h)
{
  torch::Tensor result = image;
  bool squeeze(false);
  if (result.sizes().size() == 3)
  {
    result = result.unsqueeze(0);
    squeeze = true;
  }
  torch::nn::functional::InterpolateFuncOptions options;
  options.size(std::vector<int64_t>({h, w})).mode(torch::kNearest);
  result = torch::nn::functional::interpolate(result, options);
  if (squeeze)
    result = result[0];
  return result;
}

torch::Tensor exportPreprocessedToSDL(torch::Tensor t, unsigned int w, unsigned int h)
{
  if (t.sizes()[1] != h || t.sizes()[2] != w)
    t = resizePreprocessedImage(t, w, h);
  torch::Tensor t2 = torch::zeros({4, t.sizes()[1], t.sizes()[2]});
  t2[1].copy_(t[2]);
  t2[2].copy_(t[1]);
  t2[3].copy_(t[0]);
  t2[3].sub_(-0.485/0.229).div_(1/0.229);
  t2[2].sub_(-0.456/0.224).div_(1/0.224);
  t2[1].sub_(-0.406/0.225).div_(1/0.225);
  t2.mul_(255);
  t2 = t2.transpose(2, 1);
  t2 = t2.transpose(2, 0);
  t2 = t2.contiguous();
  return t2.to(caffe2::TypeMeta::Make<unsigned char>()); // Ensure continious memory
}
