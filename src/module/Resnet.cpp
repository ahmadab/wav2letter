/*******************************************************
 * Copyright (c) 2019, Voicea
 * All rights reserved.
 *
 ********************************************************/
#include <stdexcept>

#include "module/Resnet.h"

using namespace fl;

namespace fl {

SELU::SELU() {
  alpha = 1.6732632423543772848170429916717;
  scale = 1.0507009873554804934193349852946;
}

Variable SELU::forward(const Variable& input) {
  auto mask = input >= 0.0;
  return scale * ((mask * input) + (!mask * alpha * (exp(input) - 1)));
}

std::string SELU::prettyString() const {
  return ("SELU");
}
}

namespace w2l {

//*************************************************************************************************
ResnetBasicBlock::ResnetBasicBlock(const char * activation, bool use_bn, int inplanes, int planes, int kx, int ky,
		bool downsample_x, bool downsample_y) {
	activation_.assign(activation);
	use_bn_ = use_bn;
	inplanes_ = inplanes;
	planes_ = planes;
	kx_ = kx;
	ky_ = ky;
	downsample_x_ = downsample_x;
	downsample_y_ = downsample_y;
	int stride_x = 1;
	int stride_y = 1;

	if (downsample_x_)
		stride_x = 2;
	if (downsample_y_)
		stride_y = 2;

	// BatchNorm is not implemented yet. And probably should not be implemented.

  Sequential conv;
  conv.add(Conv2D(inplanes, planes, kx, ky, stride_x, stride_y, -1, -1));
  if ((activation == "s") || (activation == "S"))
  	conv.add(SELU());
  else if ((activation == "r") || (activation == "R"))
  	conv.add(ReLU());
  else
  	throw std::invalid_argument("Unknown activation");

  conv.add(Conv2D(planes, planes, kx, ky, 1, 1, -1, -1));

  auto last_activation = ReLU();
  auto downsample_layer = Conv2D(inplanes, planes, 1, 1, stride_x, stride_y, -1, -1);
  if ((activation == "s") || (activation == "S"))
  	auto last_activation = SELU();
  else if ((activation == "r") || (activation == "R"))
  	auto last_activation = ReLU();
  else
  	throw std::invalid_argument("Unknown activation");

  add(conv);
  add(downsample_layer);
  add(last_activation);
}

std::vector<fl::Variable> ResnetBasicBlock::forward(const std::vector<fl::Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("ResnetBasicBlock module expects only one input");
  }
  return {forward(inputs[0])};
}

fl::Variable ResnetBasicBlock::forward(const fl::Variable& input) {
  auto conv_out = module(0)->forward({input})[0];
  auto downsample_out = module(1)->forward({input})[0];
  auto merge = conv_out + downsample_out;
  auto out = module(2)->forward({merge})[0];
  return out;
}

std::string ResnetBasicBlock::prettyString() const {
  std::ostringstream ss;
  ss << "Resnet Basic Block (";
  ss << inplanes_ << "->" << planes_ << "), downsample (" << std::boolalpha << downsample_x_ << ", " << downsample_y_ << ")";
  return ss.str();
}
//*************************************************************************************************


//*************************************************************************************************

ResnetSelu1D::ResnetSelu1D(int size, int inplanes) {
	size_ = size;
	inplanes_ = inplanes;
  if (size_ != 18) {
    throw std::invalid_argument("ResnetSelu1D module supports only resnet 18 based");
  }
	add(ResnetBasicBlock("s", false, inplanes_, inplanes_, 3, 1, false, false));
	add(ResnetBasicBlock("s", false, inplanes_, 2*inplanes_, 3, 1, false, false));
	add(ResnetBasicBlock("s", false, 2*inplanes_, 4*inplanes_, 3, 1, false, false));
	add(ResnetBasicBlock("s", false, 4*inplanes_, 8*inplanes_, 3, 1, false, false));
}

std::string ResnetSelu1D::prettyString() const {
  std::ostringstream ss;
  ss << "ResnetSelu1D (";
  ss << 64 << "->" << 512 << ", downsample: none)";

  for (int idx = 0; idx < modules_.size(); ++idx) {
    ss << "\n\t\t" << idx << ": ";
    ss << modules_[idx]->prettyString();
  }

  return ss.str();
}


//*************************************************************************************************

} // namespace w2l
