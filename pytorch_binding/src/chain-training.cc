#include "chain-training.h"
#include "chain-denominator.h"

namespace chain {

BaseFloat ComputeObjfAndDeriv(const ChainTrainingOptions &opts,
                    const DenominatorGraph &den_graph,
                    int32 num_sequences,
                    at::Tensor nnet_output,
                    at::Tensor nnet_output_deriv) {
  DenominatorComputation denominator(opts, den_graph, num_sequences, nnet_output);
  denominator.Forward();
  BaseFloat objf = denominator.Backward(1.0, nnet_output_deriv);
  return objf;
}

}
