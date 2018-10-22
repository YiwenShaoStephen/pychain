#include "chain-training.h"
#include "chain-denominator.h"

namespace chain {

std::vector<at::Tensor> ComputeObjfAndDeriv(const ChainTrainingOptions &opts,
					    const DenominatorGraph &den_graph,
					    int32 num_sequences,
					    at::Tensor nnet_output) {
  DenominatorComputation denominator(opts, den_graph, num_sequences, nnet_output);
  BaseFloat obj = denominator.Forward();
  at::Tensor obj_t = torch::CPU_OR_CUDA(at::kFloat).scalarTensor(obj);
  at::Tensor nnet_output_deriv = denominator.Backward();
  return {obj_t, nnet_output_deriv};
}

}
