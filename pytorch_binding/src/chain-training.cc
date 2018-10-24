#include "chain-training.h"
#include "chain-denominator.h"

namespace chain {

at::Tensor ComputeObjfAndDeriv(const ChainTrainingOptions &opts,
			       const DenominatorGraph &den_graph,
			       int32 num_sequences,
			       at::Tensor nnet_output,
			       at::Tensor nnet_output_deriv) {
  DenominatorComputation denominator(opts, den_graph, num_sequences, nnet_output);
  at::Tensor obj = denominator.Forward();
  denominator.Backward(nnet_output_deriv);
  return obj;
}

}
