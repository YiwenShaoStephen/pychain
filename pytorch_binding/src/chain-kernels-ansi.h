// chain/chain-kernels-ansi.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
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


#ifndef KALDI_CHAIN_CHAIN_KERNELS_ANSI_H_
#define KALDI_CHAIN_CHAIN_KERNELS_ANSI_H_
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "base.h"

enum { kThresholdingPowerOfTwo = 14 };

#if HAVE_CUDA == 1
//extern "C" {
  
  template <typename scalar_t>
  void cuda_chain_hmm_backward(dim3 Gr, dim3 Bl,
                               const int32_cuda *forward_transition_indices,
                               const int32_cuda *forward_transitions,
			       const scalar_t *forward_transition_probs,
                               int32_cuda num_sequences,
                               int32_cuda num_hmm_states,
                               const scalar_t *probs,
                               int32_cuda prob_stride,
                               const scalar_t *this_alpha,
                               const scalar_t *next_beta,
                               scalar_t *this_beta,
                               scalar_t *log_prob_deriv,
                               int32_cuda log_prob_deriv_stride);

  template <typename scalar_t>
  void cuda_chain_hmm_forward(dim3 Gr, dim3 Bl,
                              const int32_cuda *backward_transition_indices,
                              const int32_cuda *backward_transitions,
			      const scalar_t *backward_transition_probs,
                              int32_cuda num_sequences,
                              int32_cuda num_hmm_states,
                              const scalar_t *probs,
                              int32_cuda prob_stride,
                              const scalar_t *prev_alpha,
                              scalar_t *this_alpha);

//} // extern "C"

#endif  // HAVE_CUDA


#endif  // KALDI_CHAIN_CHAIN_KERNELS_ANSI_H_
