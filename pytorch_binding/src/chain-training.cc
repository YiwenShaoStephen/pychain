#include "chain-training.h"
#include "chain-denominator.h"

namespace chain {

torch::Tensor ComputeObjfAndDeriv(const ChainTrainingOptions &opts,
				  const DenominatorGraph &den_graph,
				  int32 num_sequences,
				  torch::Tensor nnet_output,
				  torch::Tensor nnet_output_deriv) {
  DenominatorComputation denominator(opts, den_graph, num_sequences, nnet_output);
  auto obj = denominator.Forward();
  denominator.Backward(nnet_output_deriv);
  return obj;
}

}
