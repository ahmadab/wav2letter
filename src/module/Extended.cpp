/*******************************************************
 * Copyright (c) 2019, Voicea
 * All rights reserved.
 *
 ********************************************************/
#include <stdexcept>

#include "module/Extended.h"

using namespace fl;

namespace w2l {

ExtractDim::ExtractDim(int dim, int index)
    : dim_(dim), index_(index) {}

std::vector<Variable> ExtractDim::forward(const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("ExtractDim module expects only one input");
  }
  return {forward(inputs[0])};
}

Variable ExtractDim::forward(const Variable& input) {
	//auto ndims = input.ndims();
  Variable output = input(af::span, -1, af::span, af::span);
  return output;
}

std::string ExtractDim::prettyString() const {
  std::ostringstream ss;
  ss << "Extract dimension layer: dim " << dim_ << ", index " << index_;
  return ss.str();
}


RNNOutput::RNNOutput(int idx) : idx_(idx) {};

std::vector<Variable> RNNOutput::forward(const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("RNNOutput module expects only one input");
  }
  return {forward(inputs[0])};
}

Variable RNNOutput::forward(const Variable& input) {
  Variable output = input(af::span, af::span, idx_, 0);
  return output;
}

std::string RNNOutput::prettyString() const {
  std::ostringstream ss;
  ss << "Last output from RNN";
  return ss.str();
}

} // namespace w2l
