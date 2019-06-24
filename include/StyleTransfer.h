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
    _features5_1 = x;
    return x;
  }

  torch::Tensor gram(torch::Tensor const &features)
  {
    torch::Tensor view = features[0].view({features.sizes()[1], -1});
    return torch::mm(view, view.t());
  }

  void setStyle(torch::Tensor input)
  {
    forward(input);

    _gram1_1 = gram(_features1_1);
    _gram2_1 = gram(_features2_1);
    _gram3_1 = gram(_features3_1);
    _gram4_1 = gram(_features4_1);
    _gram5_1 = gram(_features5_1);
  }

  torch::Tensor optimise(torch::Tensor &canvas)
  {
    canvas.set_requires_grad(true);
    torch::optim::Adam optim(std::vector<torch::Tensor>({canvas}), torch::optim::AdamOptions(0.05));
    for (unsigned int i(0) ; i < 1000 ; ++i)
      {
	optim.zero_grad();
	forward(canvas);
	auto loss = torch::mse_loss(gram(_features1_1), _gram1_1);
	loss += torch::mse_loss(gram(_features2_1), _gram2_1);
	loss += torch::mse_loss(gram(_features3_1), _gram3_1);
	loss += torch::mse_loss(gram(_features4_1), _gram4_1);
	loss += torch::mse_loss(gram(_features5_1), _gram5_1);
	std::cout << i << " -- " << loss.item<float>() << std::endl;
	loss.backward();
	optim.step();
      }
    return canvas;
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
  torch::Tensor _features5_1;

  torch::Tensor _gram1_1;
  torch::Tensor _gram2_1;
  torch::Tensor _gram3_1;
  torch::Tensor _gram4_1;
  torch::Tensor _gram5_1;
};

TORCH_MODULE(StyleTransfer);
