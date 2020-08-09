// Copyright       2019 Yiwen Shao
//                 2020 Yiming Wang

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include <torch/extension.h>
#include <math.h>
#include "chain-computation.h"
#include "chain-log-domain-computation.h"
#include "base.h"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::vector<torch::Tensor> ForwardBackward(
    torch::Tensor forward_transitions,
    torch::Tensor forward_transition_indices,
    torch::Tensor forward_transition_probs,
    torch::Tensor backward_transitions,
    torch::Tensor backward_transition_indices,
    torch::Tensor backward_transition_probs,
    torch::Tensor leaky_probs,
    torch::Tensor initial_probs,
    torch::Tensor final_probs,
    torch::Tensor start_state,
    torch::Tensor exp_nnet_output,
    torch::Tensor batch_sizes,
    torch::Tensor sequence_lengths,
    int num_states,
    float leaky_hmm_coefficient=1.0e-05) {
  CHECK_CONTIGUOUS(forward_transitions);
  CHECK_CONTIGUOUS(forward_transition_indices);
  CHECK_CONTIGUOUS(forward_transition_probs);
  CHECK_CONTIGUOUS(backward_transitions);
  CHECK_CONTIGUOUS(backward_transition_indices);
  CHECK_CONTIGUOUS(backward_transition_probs);
  CHECK_CONTIGUOUS(leaky_probs);
  CHECK_CONTIGUOUS(exp_nnet_output);
  CHECK_CONTIGUOUS(batch_sizes);
  CHECK_CONTIGUOUS(sequence_lengths);
  CHECK_CONTIGUOUS(initial_probs);
  CHECK_CONTIGUOUS(final_probs);
  CHECK_CONTIGUOUS(start_state);

  ChainComputation chain(
      forward_transitions,
      forward_transition_indices,
      forward_transition_probs,
      backward_transitions,
      backward_transition_indices,
      backward_transition_probs,
      leaky_probs,
      initial_probs,
      final_probs,
      start_state,
      exp_nnet_output,
      batch_sizes,
      sequence_lengths,
      num_states,
      leaky_hmm_coefficient);

  auto obj = chain.Forward();
  bool ret = chain.Backward();
  auto nnet_output_grad = chain.GetNnetGrad();
  torch::Tensor ok = torch::full({1}, ret, torch::kBool);

  return {obj, nnet_output_grad, ok};
}

std::vector<torch::Tensor> ForwardBackwardLogDomain(
    torch::Tensor forward_transitions,
    torch::Tensor forward_transition_indices,
    torch::Tensor forward_transition_probs,
    torch::Tensor backward_transitions,
    torch::Tensor backward_transition_indices,
    torch::Tensor backward_transition_probs,
    torch::Tensor initial_probs,
    torch::Tensor final_probs,
    torch::Tensor start_state,
    torch::Tensor nnet_output,
    torch::Tensor batch_sizes,
    torch::Tensor sequence_lengths,
    int num_states) {
  CHECK_CONTIGUOUS(forward_transitions);
  CHECK_CONTIGUOUS(forward_transition_indices);
  CHECK_CONTIGUOUS(forward_transition_probs);
  CHECK_CONTIGUOUS(backward_transitions);
  CHECK_CONTIGUOUS(backward_transition_indices);
  CHECK_CONTIGUOUS(backward_transition_probs);
  CHECK_CONTIGUOUS(nnet_output);
  CHECK_CONTIGUOUS(batch_sizes);
  CHECK_CONTIGUOUS(sequence_lengths);
  CHECK_CONTIGUOUS(initial_probs);
  CHECK_CONTIGUOUS(final_probs);
  CHECK_CONTIGUOUS(start_state);

  ChainLogDomainComputation chain(
      forward_transitions,
      forward_transition_indices,
      forward_transition_probs,
      backward_transitions,
      backward_transition_indices,
      backward_transition_probs,
      initial_probs,
      final_probs,
      start_state,
      nnet_output,
      batch_sizes,
      sequence_lengths,
      num_states);

  auto obj = chain.Forward();
  bool ret = chain.Backward();
  auto nnet_output_grad = chain.GetNnetGrad();
  torch::Tensor ok = torch::full({1}, ret, torch::kBool);

  return {obj, nnet_output_grad, ok};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_backward", &ForwardBackward);
  m.def("forward_backward_log_domain", &ForwardBackwardLogDomain);
  m.def("set_verbose_level", &SetVerboseLevel);
}
