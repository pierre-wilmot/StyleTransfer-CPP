#pragma once

#include <queue>
#include <torch/torch.h>

#include "TextureModel.h"

class StyleTransferDelegate
{
public:
  virtual void onUpdate(torch::Tensor t) = 0;
  virtual void onFinished(torch::Tensor t) = 0;
};

class StyleTransferImpl : public torch::nn::Module
{
public:
  StyleTransferImpl()
    :_conv1_1(register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv1_2(register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv2_1(register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv2_2(register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv3_1(register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv3_2(register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv3_3(register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv3_4(register_module("conv3_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv4_1(register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv4_2(register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv4_3(register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv4_4(register_module("conv4_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding({1, 1}).padding_mode(torch::kCircular))))
    ,_conv5_1(register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding({1, 1}).padding_mode(torch::kCircular))))
  {
    // Freezing the network
    for (auto & p : parameters(true))
      p.set_requires_grad(false);
  }

  void setDelegate(StyleTransferDelegate *d)
  {
    _delegate = d;
  }

  void stopOptmising()
  {
    _keepOptimising = false;
  }

  bool checkInputSize(torch::Tensor const &x) const
  {
    return !(x.sizes()[2] < 3 || x.sizes()[3] < 3);
  }

  torch::Tensor forward(torch::Tensor input)
  {
    _features1_1 = torch::Tensor();
    _features2_1 = torch::Tensor();
    _features3_1 = torch::Tensor();
    _features4_1 = torch::Tensor();
    _features5_1 = torch::Tensor();

    torch::Tensor x = input;
    if (x.dim() == 3)
      x = x.unsqueeze(0);
    x = torch::relu(_conv1_1(x));
    _features1_1 = x;
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv1_2(x));
    if (!checkInputSize(x)) return x;
    x = torch::max_pool2d(x, {2, 2});
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv2_1(x));
    _features2_1 = x;
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv2_2(x));
    if (!checkInputSize(x)) return x;
    x = torch::max_pool2d(x, {2, 2});
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv3_1(x));
    _features3_1 = x;
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv3_2(x));
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv3_3(x));
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv3_4(x));
    if (!checkInputSize(x)) return x;
    x = torch::max_pool2d(x, {2, 2});
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv4_1(x));
    _features4_1 = x;
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv4_2(x));
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv4_3(x));
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv4_4(x));
    if (!checkInputSize(x)) return x;
    x = torch::max_pool2d(x, {2, 2});
    if (!checkInputSize(x)) return x;
    x = torch::relu(_conv5_1(x));
    _features5_1 = x;
    return x;
  }

  torch::Tensor gram(torch::Tensor const &features) const
  {
    // Do not compute a GRAM matrices for feature maps smaller than 9x9
    // This is to avoid reproducing the input strcture too closely
    if (!features.defined() || features.sizes()[2] < 9 || features.sizes()[3] < 9)
      return torch::Tensor();

    torch::Tensor view = features[0].view({features.sizes()[1], -1});
    torch::Tensor gram = torch::mm(view, view.t());
    // Divide the gram by the number of elements that went in.
    // This is so that the values are in the same range if style and canvas have different resolutions
    gram /= view.sizes()[1];
    return gram;
  }

  torch::Tensor histogram(torch::Tensor const &features)
  {
    if (!features.defined())
      return torch::Tensor();
    int64_t b = features.size(0);
    int64_t c = features.size(1);
    return std::get<0>(torch::sort(features.view(std::vector<int64_t>({b, c, -1})), 2)).view(features.sizes());
  }

  void setPaddingMode(const torch::nn::detail::conv_padding_mode_t &new_padding_mode, int padding)
  {
    _conv1_1->options.padding_mode(new_padding_mode).padding(padding);
    _conv1_2->options.padding_mode(new_padding_mode).padding(padding);
    _conv2_1->options.padding_mode(new_padding_mode).padding(padding);
    _conv2_2->options.padding_mode(new_padding_mode).padding(padding);
    _conv3_1->options.padding_mode(new_padding_mode).padding(padding);
    _conv3_2->options.padding_mode(new_padding_mode).padding(padding);
    _conv3_3->options.padding_mode(new_padding_mode).padding(padding);
    _conv3_4->options.padding_mode(new_padding_mode).padding(padding);
    _conv4_1->options.padding_mode(new_padding_mode).padding(padding);
    _conv4_2->options.padding_mode(new_padding_mode).padding(padding);
    _conv4_3->options.padding_mode(new_padding_mode).padding(padding);
    _conv4_4->options.padding_mode(new_padding_mode).padding(padding);
    _conv5_1->options.padding_mode(new_padding_mode).padding(padding);
  }

  void setCircularPadding()
  {
    setPaddingMode(torch::kCircular, 1);
  }

  void setNoPadding(int v)
  {
    setPaddingMode(torch::kZeros, v);
  }

  virtual void setStyle(torch::Tensor input)
  {
    setNoPadding(0);
    forward(input);
    _model.gram1_1 = gram(_features1_1);
    _model.gram2_1 = gram(_features2_1);
    _model.gram3_1 = gram(_features3_1);
    _model.gram4_1 = gram(_features4_1);
    _model.gram5_1 = gram(_features5_1);

    _model.style1_1 = histogram(_features1_1);
    _model.style2_1 = histogram(_features2_1);
    _model.style3_1 = histogram(_features3_1);
    _model.style4_1 = histogram(_features4_1);
    _model.style5_1 = histogram(_features5_1);
  }

  virtual void setContent(torch::Tensor input)
  {
    setNoPadding(1);
    forward(input);
    _model.content = _features4_1.clone();
  }

  void setModel(TextureModel const &model)
  {
    _model = model;
  }

  TextureModel getModel() const
  {
    return _model.clone();
  }

  torch::Tensor gramLoss(torch::Tensor const &features, torch::Tensor const &targetGram)
  {
    torch::Tensor g = gram(features);
    if (g.defined() && targetGram.defined())
      return torch::mse_loss(g, targetGram);
    return at::zeros(std::vector<int64_t>({}));
  }

  torch::Tensor histogramLoss(torch::Tensor const &x, torch::Tensor const &y)
  {
    if (!x.defined() || !y.defined())
      return at::zeros(std::vector<int64_t>({}));
    torch::nn::functional::InterpolateFuncOptions options;
    options.size(std::vector<int64_t>({y.size(2), y.size(3)})).mode(torch::kNearest);
    torch::Tensor t = torch::nn::functional::interpolate(x, options);
    int64_t b = t.size(0);
    int64_t c = t.size(1);
    t = std::get<0>(torch::sort(t.view(std::vector<int64_t>({b, c, -1})), 2));
    return torch::mse_loss(t, y.view(std::vector<int64_t>({b, c, -1})));
  }

  virtual torch::Tensor computeLoss(torch::Tensor &canvas)
  {
    forward(canvas);

    torch::Tensor loss = torch::mse_loss(gram(_features1_1), _model.gram1_1);
    loss += gramLoss(_features2_1, _model.gram2_1);
    loss += gramLoss(_features3_1, _model.gram3_1);
    loss += gramLoss(_features4_1, _model.gram4_1);
    loss += gramLoss(_features5_1, _model.gram5_1);

    torch::Tensor histoLoss = histogramLoss(_features1_1, _model.style1_1);
    histoLoss += histogramLoss(_features2_1, _model.style2_1);
    histoLoss += histogramLoss(_features3_1, _model.style3_1);
    histoLoss += histogramLoss(_features4_1, _model.style4_1);
    histoLoss += histogramLoss(_features5_1, _model.style5_1);

    if (_model.content.defined())
      loss += torch::mse_loss(_features4_1, _model.content) / 5;

    loss += histoLoss / 100000;

    return loss;
  }

  torch::Tensor optimise(torch::Tensor &canvas)
  {
    canvas.set_requires_grad(true);
    torch::optim::Adam optim(std::vector<torch::Tensor>({canvas}), torch::optim::AdamOptions(0.01));
    std::queue<float> losses;
    unsigned int i(0);
    _keepOptimising = true;
    setCircularPadding();
    while (_keepOptimising)
      {
	optim.zero_grad();
	torch::Tensor loss = computeLoss(canvas);
	std::cout << canvas.sizes()[2] << " - " << i << " -- " << loss.item<float>() << std::endl;
	i++;
	if (i > 100 && losses.front() < losses.back())
	  break;
	losses.push(loss.item<float>());
	if (losses.size() > 10)
	  losses.pop();
	loss.backward();
	optim.step();
	if (_delegate && i % 10 == 0)
	  _delegate->onUpdate(canvas.clone());
      }
    canvas.set_requires_grad(false);
    if (_delegate)
      _delegate->onFinished(canvas.clone());
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

  TextureModel _model;

  StyleTransferDelegate *_delegate = nullptr;
  bool _keepOptimising;
};

TORCH_MODULE(StyleTransfer);
