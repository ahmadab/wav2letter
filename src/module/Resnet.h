/*******************************************************
 * Copyright (c) 2019, Voicea
 * All rights reserved.
 *
 ********************************************************/
#pragma once

#include <cereal/types/set.hpp>
#include <cereal/types/unordered_set.hpp>

#include <flashlight/flashlight.h>

namespace fl {
class SELU : public UnaryModule {
 private:
  double alpha;
  double scale;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule)

 public:
  SELU();

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};
}

namespace w2l {


class ResnetBasicBlock : public fl::Container {
 private:
	ResnetBasicBlock() = default;
	std::string activation_;
	bool use_bn_;
	int inplanes_;
	int planes_;
	int kx_;
	int ky_;
	bool downsample_x_;
	bool downsample_y_;
  FL_SAVE_LOAD_WITH_BASE(fl::Container, activation_, use_bn_, inplanes_, planes_, kx_, ky_, downsample_x_, downsample_y_)

 public:
  explicit ResnetBasicBlock(const char * activation, bool use_bn, int inplanes, int planes, int kx, int ky, bool downsample_x, bool downsample_y);

  std::vector<fl::Variable> forward(const std::vector<fl::Variable>& inputs) override;

  fl::Variable forward(const fl::Variable& input);

  std::string prettyString() const override;
};

class ResnetSelu1D : public fl::Sequential {
 private:
	ResnetSelu1D() = default;
	int size_;
	int inplanes_;
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential, size_, inplanes_)

 public:
  explicit ResnetSelu1D(int size, int inplanes);

//  std::vector<fl::Variable> forward(const std::vector<fl::Variable>& inputs) override;

//  fl::Variable forward(const fl::Variable& input);

  std::string prettyString() const override;
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::ResnetBasicBlock);
CEREAL_REGISTER_TYPE(w2l::ResnetSelu1D);
CEREAL_REGISTER_TYPE(fl::SELU);

