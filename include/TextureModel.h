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
  torch::Tensor style1_1;
  torch::Tensor style2_1;
  torch::Tensor style3_1;
  torch::Tensor style4_1;
  torch::Tensor style5_1;

  TextureModel clone() const
  {
    TextureModel clone;
    if (gram1_1.defined()) clone.gram1_1 = gram1_1.clone();
    if (gram2_1.defined()) clone.gram2_1 = gram2_1.clone();
    if (gram3_1.defined()) clone.gram3_1 = gram3_1.clone();
    if (gram4_1.defined()) clone.gram4_1 = gram4_1.clone();
    if (gram5_1.defined()) clone.gram5_1 = gram5_1.clone();
    if (content.defined()) clone.content = content.clone();
    if (style1_1.defined()) clone.style1_1 = style1_1.clone();
    if (style2_1.defined()) clone.style2_1 = style2_1.clone();
    if (style3_1.defined()) clone.style3_1 = style3_1.clone();
    if (style4_1.defined()) clone.style4_1 = style4_1.clone();
    if (style5_1.defined()) clone.style5_1 = style5_1.clone();
    return clone;
  }
};
