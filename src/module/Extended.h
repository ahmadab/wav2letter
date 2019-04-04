/*******************************************************
 * Copyright (c) 2019, Voicea
 * All rights reserved.
 *
 ********************************************************/
#pragma once

#include <cereal/types/set.hpp>
#include <cereal/types/unordered_set.hpp>

#include <flashlight/flashlight.h>

namespace w2l {

/**
 * A module for creating a extended functions and modules.
 */
class ExtractDim : public fl::Container {
 private:
	ExtractDim() = default; // Intentionally private
	int dim_;
	int index_;
  FL_SAVE_LOAD_WITH_BASE(fl::Container, dim_, index_)

 public:
  explicit ExtractDim(int dim, int index);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  fl::Variable forward(const fl::Variable& input);

  std::string prettyString() const override;
};



/**
 * A module that get output from RNN
 */
class RNNOutput : public fl::Container {
 private:
	RNNOutput() = default; // Intentionally private
	int idx_;
  FL_SAVE_LOAD_WITH_BASE(fl::Container, idx_)

 public:
  explicit RNNOutput(int idx);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  fl::Variable forward(const fl::Variable& input);

  std::string prettyString() const override;
};


} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::ExtractDim);
CEREAL_REGISTER_TYPE(w2l::RNNOutput);
