
#include <iomanip>
#include <iostream>

#include <arrayfire.h>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "common/Utils.h"
#include "module/module.h"
#include <iostream>


using namespace fl;
using namespace w2l;

Sequential model;
Variable input2, gradoutput, b;


std::string print_summary(std::shared_ptr<fl::Sequential> model, af::array &input) {
	  std::ostringstream ss;
	  af::dim4 iDims = input.dims();
	  ss << "Sequential";
	  ss << " [input:" << iDims;
	  auto x = noGrad(input);
	  std::vector<Variable> output = {x};
	  for (int i = 0; i < model->modules().size(); ++i) {
	  	auto input_dims = output.front().dims();
	  	output = model->module(i)->forward(output);
	    ss << "\n\t(" << i << "): input dims: " << input_dims << ", output dims: " << output.front().dims() << ", " << model->module(i)->prettyString();
	  }
	  ss << "\n]\n";
	  return ss.str();
}

int main(int argc, char** argv) {

  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
	argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
	  "Usage: \n " + exec + " train [flags]\n or " + std::string() +
	  " continue [directory] [flags]\n or " + std::string(argv[0]) +
	  " fork [directory/model] [flags]");

  /* ===================== Parse Options ===================== */
  std::string archfile = argv[1];
  int length = std::stoi(argv[2]);
  if (argc <= 1) {
	LOG(FATAL) << gflags::ProgramUsage();
  }

  af::info();

  int nchannel = 1;
	int nclass = 40;
	int batchsize = 8;

	auto model = createW2lSeqModule(archfile, nchannel, nclass);

  auto input = af::randn(length, 1, nchannel, batchsize, f32);
	//auto prettystr = model->prettyString();
  auto prettystr = print_summary(model, input);
  std::cout << prettystr;

}
