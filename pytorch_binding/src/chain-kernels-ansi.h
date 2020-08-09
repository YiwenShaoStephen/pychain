// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2019  Yiwen Shao

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <cuda.h>
#include <cuda_runtime.h>
#include "base.h"

enum { kThresholdingPowerOfTwo = 14 };

extern "C" {
  void cuda_chain_hmm_forward(dim3 Gr, dim3 Bl,
                              const int *backward_transition_indices,
                              const int *backward_transitions,
                              const float *backward_transition_probs,
                              const float *probs,
                              float *alpha,
                              int t,
                              int num_sequences,
                              int num_frames,
                              int num_hmm_states,
                              int num_pdfs,
                              int num_transitions);

  void cuda_chain_hmm_backward(dim3 Gr, dim3 Bl,
                               const int *forward_transition_indices,
                               const int *forward_transitions,
                               const float *forward_transition_probs,
                               const float *probs,
                               const float *alpha,
                               float *beta,
                               float *log_prob_deriv,
                               int t,
                               int num_sequences,
                               int num_frames,
                               int num_hmm_states,
                               int num_pdfs,
                               int num_transitions);

  void cuda_chain_hmm_log_domain_forward(dim3 Gr, dim3 Bl,
                                         const int *backward_transition_indices,
                                         const int *backward_transitions,
                                         const float *backward_transition_probs,
                                         const float *probs,
                                         float *alpha,
                                         int t,
                                         int num_sequences,
                                         int num_frames,
                                         int num_hmm_states,
                                         int num_pdfs,
                                         int num_transitions);

  void cuda_chain_hmm_log_domain_backward(dim3 Gr, dim3 Bl,
                                          const int *forward_transition_indices,
                                          const int *forward_transitions,
                                          const float *forward_transition_probs,
                                          const float *probs,
                                          const float *alpha,
                                          float *beta,
                                          float *log_prob_deriv,
                                          int t,
                                          int num_sequences,
                                          int num_frames,
                                          int num_hmm_states,
                                          int num_pdfs,
                                          int num_transitions);


} // extern "C"
