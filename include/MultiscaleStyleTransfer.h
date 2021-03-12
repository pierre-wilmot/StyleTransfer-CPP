#pragma once

#include "StyleTransfer.h"

class MultiscaleStyleTransferImpl : public StyleTransferImpl
{
public:
  MultiscaleStyleTransferImpl(unsigned int scales)
    :_scales(scales), _models(scales)
  {
  }

  void setStyle(torch::Tensor input) override
  {
    torch::nn::functional::InterpolateFuncOptions options;
    options.scale_factor(std::vector<double>({.5, .5})).mode(torch::kBicubic).align_corners(false).recompute_scale_factor(false);
    torch::Tensor style = input;
    for (unsigned int i(0); i < _scales ; ++i)
    {
      setModel(_models[i]);
      StyleTransferImpl::setStyle(style);
      _models[i] = getModel();
      style = torch::nn::functional::interpolate(style.unsqueeze(0), options)[0];
    }
  }

  void setContent(torch::Tensor input) override
  {
    torch::nn::functional::InterpolateFuncOptions options;
    options.scale_factor(std::vector<double>({.5, .5})).mode(torch::kBicubic).align_corners(false).recompute_scale_factor(false);
    torch::Tensor content = input;
    for (unsigned int i(0); i < _scales ; ++i)
    {
      setModel(_models[i]);
      StyleTransferImpl::setContent(content);
      _models[i] = getModel();
      content = torch::nn::functional::interpolate(content.unsqueeze(0), options)[0];
    }
  }

  torch::Tensor computeLoss(torch::Tensor &input) override
  {
    torch::nn::functional::InterpolateFuncOptions options;
    options.scale_factor(std::vector<double>({.5, .5})).mode(torch::kBicubic).align_corners(false).recompute_scale_factor(false);;
    torch::Tensor x = input;
    torch::Tensor multiscale_loss;
    for (unsigned int i(0); i < _scales ; ++i)
    {
      setModel(_models[i]);
      torch::Tensor loss = StyleTransferImpl::computeLoss(x);
      multiscale_loss = multiscale_loss.defined() ? multiscale_loss + loss : loss;
      x = torch::nn::functional::interpolate(x.unsqueeze(0), options)[0];
      if (x.sizes()[1] < 9 || x.sizes()[2] < 9 )
	break;
    }
    return multiscale_loss;
  }


private:
  unsigned int _scales;
  std::vector<TextureModel> _models;
};
TORCH_MODULE(MultiscaleStyleTransfer);
