// Copyright  2015  Johns Hopkins University (author: Daniel Povey)
//            2019  Yiwen Shao
//            2020  Yiming Wang

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


#include <cfloat>
#include "chain-kernels-ansi.h"
#include <stdio.h>

#ifdef __CUDACC__
#if ( __CUDACC_VER_MAJOR__ >= 8 ) && ( !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 )
// native implementation available
#else
#if __CUDA_ARCH__ >= 600
#error using CAS implementation of double atomicAdd
#endif
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif
#endif


template <typename Real>
__device__ inline void atomic_add(Real* address, Real value) {
  atomicAdd(address, value);
}

template <typename Real>
__device__ inline void atomic_add_thresholded(Real* address, Real value) {
  // This function uses a randomized algorithm to only do atomic adds for values
  // >=n a threshold, and if it's below the threshold, randomly add the
  // threshold itself with probability (value / threshold).  This preserves
  // expectations.  Note: we assume that value >= 0.

  // kThresholdingPowerOfTwo defines
  // the threshold for randomized posterior pruning.
  const Real threshold = 1.0 / (1 << kThresholdingPowerOfTwo);
  if (value >= threshold) {
    atomic_add(address, value);
  } else {
    // The intention here is to do:
    // with probability(value / threshold), do:
    //   atomic_add(address, threshold);
    // We use the least significant bits of the value as a source of
    // randomness.  It would probably be more efficient to extract these
    // random bits directly from the float, but I don't want to have to
    // deal with endian-ness issues.
    //
    // below, x is a fixed-point representation of (value / threshold); it would
    // be 16777216 == 2^24 if value == threshold and 0 if value == 0.  We choose
    // the power 24 because that's the number of binary digits in the mantissa
    // in IEEE single precision floating point.
    // Note: we parenthesize the expression like this so that the
    // denominator can be precomputed as a constant expression.
    int x = value / (threshold / (1 << 24));
    // in the line below, the expression (x >> 12) is a representation of (value /
    // threshold) between 0 and 4096, with 4096 representing (value / threshold ==
    // 1), while (x & 4095) is treated as a pseudorandom number between 0 and 4095.
    if ((x >> 12) > (x & 4095))
      atomic_add(address, threshold);
  }
}

// one iteration of the forward computation in the 'tombstone' CTC HMM computation.
// The grid y determines which HMM-state we handle.  [put this in the grid because
// HMM-states don't all take the same amount of time in the backwards direction, and it's
// better for scheduling to have them at the outer level.]
// The block x and grid x determine which sequence (0 ... num_sequences - 1) we handle;
// note that num_sequences == the number of elements in the minibatch, and we
// insist they all have the same number of time steps.

__global__
static void _cuda_chain_hmm_forward(const int *backward_transition_indices,
                                    const int *backward_transitions,
                                    const float *backward_transition_probs,
                                    const float *probs,
                                    float *alpha,
                                    int t,
                                    int num_sequences,
                                    int num_frames,
                                    int num_hmm_states,
                                    int num_pdfs,
                                    int num_transitions) {
  // s is the index of the sequence within the minibatch,
  // from 0 .. num-egs-in-this-minibatch - 1.
  // h is the hmm-state index.
  int s = threadIdx.x + blockIdx.x * blockDim.x,
    h  = blockIdx.y;
  if (s >= num_sequences)
    return;

  // T, H, D, K are used as strides
  int T = num_frames,
    H = num_hmm_states,
    D = num_pdfs,
    K = num_transitions;

  float this_tot_alpha = 0.0;
  int trans_i = backward_transition_indices[s * H * 2 + h * 2],
      trans_end = backward_transition_indices[s * H * 2 + h * 2 + 1];
  // Note: regarding this loop unrolling, I tried the automatic unrolling using
  // #pragma unroll 2 (after modifying the loop to have an integer index), but I
  // did not see any performance improvement, it was slightly slower.  So the
  // compiler must be doing something different than what I'm doing here.
  const int loop_unroll = 2;  // don't change this without changing the code
                              // below.
  for (; trans_i + loop_unroll <= trans_end; trans_i += loop_unroll) {
    float transition_prob0 = backward_transition_probs[s * K + trans_i];
    int pdf_id0 = backward_transitions[s * K * 3 + trans_i * 3 + 2],
        prev_hmm_state0 = backward_transitions[s * K * 3 + trans_i * 3];
    float transition_prob1 = backward_transition_probs[s * K + trans_i + 1];
    int pdf_id1 = backward_transitions[s * K * 3 + (trans_i + 1) * 3 + 2],
      prev_hmm_state1 = backward_transitions[s * K * 3 + (trans_i + 1) * 3];
    float pseudo_loglike0 = probs[s * T * D + (t-1) * D + pdf_id0],
      this_prev_alpha0 = alpha[s * (T+1) * (H+1) + (t-1) * (H+1) + prev_hmm_state0],
      pseudo_loglike1 = probs[s * T * D + (t-1) * D + pdf_id1],
      this_prev_alpha1 = alpha[s * (T+1) * (H+1) + (t-1) * (H+1) + prev_hmm_state1];

    this_tot_alpha += this_prev_alpha0 * transition_prob0 * pseudo_loglike0 +
                       this_prev_alpha1 * transition_prob1 * pseudo_loglike1;
  }
  if (trans_i != trans_end) {
    // mop up the odd transition.
    float transition_prob0 = backward_transition_probs[s * K + trans_i];
    int pdf_id0 = backward_transitions[s * K * 3 + trans_i * 3 + 2],
      prev_hmm_state0 = backward_transitions[s * K * 3 + trans_i * 3];
    float pseudo_loglike0 = probs[s * T * D + (t-1) * D + pdf_id0],
      this_prev_alpha0 = alpha[s * (T+1) * (H+1) + (t-1) * (H+1) + prev_hmm_state0];
    this_tot_alpha += this_prev_alpha0 * transition_prob0 * pseudo_loglike0;
  }

  // Let arbitrary_scale be the inverse of the sum of all alpha values on-- the
  // previous frame this sum of all the alpha values is stored in the place that
  // we'd store the previous alpha for state-index equal to num_hmm_states
  // (i.e. one past the end).  We multiply this into all the
  // transition-probabilities from the previous frame to this frame, in both the
  // forward and backward passes, in order to keep the alphas in a good numeric
  // range.  This won't affect the posteriors, as it's just a constant factor
  // for each frame, but when computing the total likelihood we'll need to
  // compensate for it later on.
  float arbitrary_scale =
    1.0 / alpha[s * (T+1) * (H+1) + (t-1) * (H+1) + H];
  alpha[s * (T+1) * (H+1) + t * (H+1) + h] = this_tot_alpha * arbitrary_scale;
}


__global__
static void _cuda_chain_hmm_backward(const int *forward_transition_indices,
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
                                     int num_transitions) {
  // s is the index of the sequence within the minibatch,
  // from 0 .. num-egs-in-this-minibatch - 1.
  // h is the hmm-state index.
  int s = threadIdx.x + blockIdx.x * blockDim.x,
    h = blockIdx.y;
  if (s >= num_sequences)
    return;

  // T, H, D, K are used as strides
  int T = num_frames,
    H = num_hmm_states,
    D = num_pdfs,
    K = num_transitions;

  // See where arbitrary_scale is defined in the forward computation above
  float this_alpha_prob = alpha[s * (T+1) * (H+1) + t * (H+1) + h],
    arbitrary_scale = 1.0 / alpha[s * (T+1) * (H+1)  + t * (H+1) + H];
  float tot_variable_factor = 0.0;

  float occupation_factor = this_alpha_prob * arbitrary_scale;
  int trans_i = forward_transition_indices[s * H * 2 + h * 2],
    trans_end = forward_transition_indices[s * H * 2 + h * 2 + 1];
  const int loop_unroll = 2;  // don't change this without changing the code
                              // below.
  for (; trans_i + loop_unroll <= trans_end; trans_i += loop_unroll) {
    float transition_prob0 = forward_transition_probs[s * K + trans_i];
    int pdf_id0 = forward_transitions[s * K * 3 + trans_i * 3 + 2],
      next_hmm_state0 = forward_transitions[s * K * 3 + trans_i * 3 + 1];
    float transition_prob1 = forward_transition_probs[s * K + trans_i + 1];
    int pdf_id1 = forward_transitions[s * K * 3 + (trans_i + 1) * 3 + 2],
      next_hmm_state1 = forward_transitions[s * K * 3 + (trans_i + 1) * 3 + 1];
    float variable_factor0 = transition_prob0 *
      beta[s * 2 * H + ((t+1) % 2) * H + next_hmm_state0] *
      probs[s * T * D + t * D + pdf_id0];
    float variable_factor1 = transition_prob1 *
      beta[s * 2 * H + ((t+1) % 2) * H + next_hmm_state1] *
      probs[s * T * D + t * D + pdf_id1];
    tot_variable_factor += variable_factor0 + variable_factor1;
    float occupation_prob0 = variable_factor0 * occupation_factor;
    atomic_add_thresholded(log_prob_deriv + s * T * D + t * D + pdf_id0,
                           occupation_prob0);
    float occupation_prob1 = variable_factor1 * occupation_factor;
    atomic_add_thresholded(log_prob_deriv + s * T * D + t * D + pdf_id1,
                           occupation_prob1);
  }
  if (trans_i != trans_end) {
    // mop up the odd transition.
    float transition_prob0 = forward_transition_probs[s * K + trans_i];
    int pdf_id0 = forward_transitions[s * K * 3 + trans_i * 3 + 2],
      next_hmm_state0 = forward_transitions[s * K * 3 + trans_i * 3 + 1];
    float variable_factor0 = transition_prob0 *
      beta[s * 2 * H + ((t+1) % 2) * H + next_hmm_state0] *
      probs[s * T * D + t * D + pdf_id0];
    tot_variable_factor += variable_factor0;
    float occupation_prob0 = variable_factor0 * occupation_factor;
    atomic_add_thresholded(log_prob_deriv + s * T * D + t * D + pdf_id0,
                           occupation_prob0);
  }
  beta[s * 2 * H + (t%2) * H + h] = tot_variable_factor * arbitrary_scale;
}


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
                            int num_transitions) {
  _cuda_chain_hmm_forward<<<Gr,Bl>>>(backward_transition_indices,
                                     backward_transitions,
                                     backward_transition_probs,
                                     probs,
                                     alpha,
                                     t,
                                     num_sequences,
                                     num_frames,
                                     num_hmm_states,
                                     num_pdfs,
                                     num_transitions);
}

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
                             int num_transitions) {
  _cuda_chain_hmm_backward<<<Gr,Bl>>>(forward_transition_indices,
                                      forward_transitions,
                                      forward_transition_probs,
                                      probs,
                                      alpha,
                                      beta,
                                      log_prob_deriv,
                                      t,
                                      num_sequences,
                                      num_frames,
                                      num_hmm_states,
                                      num_pdfs,
                                      num_transitions);
}
