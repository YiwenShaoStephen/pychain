#include <torch/extension.h>
#include <math.h>
#include "chain-denominator.h"
#include "base.h"

std::vector<torch::Tensor> ForwardBackwardDen(torch::Tensor forward_transitions,
					      torch::Tensor forward_transition_indices,
					      torch::Tensor forward_transition_probs,
					      torch::Tensor backward_transitions,
					      torch::Tensor backward_transition_indices,
					      torch::Tensor backward_transition_probs,
					      torch::Tensor initial_probs,
					      torch::Tensor exp_nnet_output,
					      int num_states) {
  DenominatorComputation denominator(forward_transitions,
				     forward_transition_indices,
				     forward_transition_probs,
				     backward_transitions,
				     backward_transition_indices,
				     backward_transition_probs,
				     initial_probs,
				     exp_nnet_output,
				     num_states);
  auto obj = denominator.Forward();
  denominator.Backward();
  auto alpha = denominator.GetAlpha();
  auto nnet_output_grad = denominator.GetNnetGrad();
  return {obj, nnet_output_grad, alpha};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_backward_den", &ForwardBackwardDen);
  m.def("set_verbose_level", &SetVerboseLevel);
}
