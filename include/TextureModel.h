#pragma once

#include <torch/torch.h>

struct TextureModel
{
  torch::Tensor gram1_1;
  torch::Tensor gram2_1;
  torch::Tensor gram3_1;
  torch::Tensor gram4_1;
  torch::Tensor gram5_1;
  torch::Tensor content;

  TextureModel clone() const
  {
    TextureModel clone;
    if (gram1_1.defined()) clone.gram1_1 = gram1_1.clone();
    if (gram2_1.defined()) clone.gram2_1 = gram2_1.clone();
    if (gram3_1.defined()) clone.gram3_1 = gram3_1.clone();
    if (gram4_1.defined()) clone.gram4_1 = gram4_1.clone();
    if (gram5_1.defined()) clone.gram5_1 = gram5_1.clone();
    if (content.defined()) clone.content = content.clone();
    return clone;
  }
};
