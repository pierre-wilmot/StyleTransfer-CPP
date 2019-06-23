#pragma once

#include <torch/torch.h>

class StyleTransferImpl : public torch::nn::Module
{
public:
  StyleTransferImpl()
    :_conv1_1(register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1))))
    ,_conv1_2(register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1))))
    ,_conv2_1(register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1))))
    ,_conv2_2(register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1))))
    ,_conv3_1(register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1))))
    ,_conv3_2(register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1))))
    ,_conv3_3(register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1))))
    ,_conv3_4(register_module("conv3_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1))))
    ,_conv4_1(register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1))))
    ,_conv4_2(register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1))))
    ,_conv4_3(register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1))))
    ,_conv4_4(register_module("conv4_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1))))
    ,_conv5_1(register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1))))
  {
    // Freezing the network
    for (auto & p : parameters(true))
      p.set_requires_grad(false);    
  }


  torch::Tensor forward(torch::Tensor input)
  {
    torch::Tensor x = input;
    x = torch::relu(_conv1_1(x));
    _features1_1 = x;
    x = torch::relu(_conv1_2(x));
    x = torch::max_pool2d(x, {2, 2});
    x = torch::relu(_conv2_1(x));
    _features2_1 = x;
    x = torch::relu(_conv2_2(x));
    x = torch::max_pool2d(x, {2, 2});
    x = torch::relu(_conv3_1(x));
    _features3_1 = x;
    x = torch::relu(_conv3_2(x));
    x = torch::relu(_conv3_3(x));
    x = torch::relu(_conv3_4(x));
    x = torch::max_pool2d(x, {2, 2});
    x = torch::relu(_conv4_1(x));
    _features4_1 = x;
    x = torch::relu(_conv4_2(x));
    x = torch::relu(_conv4_3(x));
    x = torch::relu(_conv4_4(x));
    x = torch::max_pool2d(x, {2, 2});
    x = torch::relu(_conv5_1(x));
    return x;
  }
  
private:  
  torch::nn::Conv2d _conv1_1;
  torch::nn::Conv2d _conv1_2;
  torch::nn::Conv2d _conv2_1;
  torch::nn::Conv2d _conv2_2;
  torch::nn::Conv2d _conv3_1;
  torch::nn::Conv2d _conv3_2;
  torch::nn::Conv2d _conv3_3;
  torch::nn::Conv2d _conv3_4;
  torch::nn::Conv2d _conv4_1;
  torch::nn::Conv2d _conv4_2;
  torch::nn::Conv2d _conv4_3;
  torch::nn::Conv2d _conv4_4;
  torch::nn::Conv2d _conv5_1; 

  torch::Tensor _features1_1;
  torch::Tensor _features2_1;
  torch::Tensor _features3_1;
  torch::Tensor _features4_1;
};

TORCH_MODULE(StyleTransfer);
