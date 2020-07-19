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


/*
  This implementation of log1p is obtained from
  https://forums.developer.nvidia.com/t/faster-and-more-accurate-implementation-of-log1pf/40575/14

  Copyright (c) 2015-2017, Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* log1p(a) = log(a+1) = log(2**e * t) = log(2)*e + log(t). With t = m + 1,
   log1p(a) = log(2)*e + log1p(m). Choose e such that m is in [-0.25, 0.5],
   with s = 2**(-e) we then have m = s*(a+1) - 1 = s*a + (s - 1). Instead
   of using s directly, an intermediate scale factor s' = 4*s is utilized
   to ensure this is representable as a normalized floating-point number.

   max ulp err = 0.87454
*/
__device__ float my_log1pf (float a)
{
    float m, r, s, t, u; 
    int e;

    u = __fadd_rz (a, 1.0f);
    e = (__float_as_int (u) - __float_as_int (0.75f)) & 0xff800000;
    m = __int_as_float (__float_as_int (a) - e);
    s = __int_as_float (__float_as_int (4.0f) - e); // s' in [2**-126,2**26]
    m = m + fmaf (0.25f, s, -1.0f);
    // approximate log(1+m) on [-0.25, 0.5]
    s = m * m;
    r =             -4.53948975e-2f;  // -0x1.73e000p-5
    t =              1.05468750e-1f;  //  0x1.b00000p-4
    r = fmaf (r, s, -1.32274792e-1f); // -0x1.0ee616p-3
    t = fmaf (t, s,  1.44911826e-1f); //  0x1.28c788p-3
    r = fmaf (r, s, -1.66412741e-1f); // -0x1.54d034p-3
    t = fmaf (t, s,  1.99887201e-1f); //  0x1.995e76p-3
    r = fmaf (r, s, -2.50002742e-1f); // -0x1.0000b8p-2
    r = fmaf (t, m, r);
    r = fmaf (r, m,  3.33335280e-1f); //  0x1.5555d8p-2
    r = fmaf (r, m, -4.99999970e-1f); // -0x1.fffffep-2
    r = fmaf (r, s, m);
    r = fmaf ((float)e, 0.693147182f * 1.1920929e-7f, r); // log(2) * 0x1.0p-23
    if (!((a != 0.0f) && (u > 0.0f) && (a < __int_as_float (0x7f800000)))) {
        asm ("lg2.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(u));
        r = __fadd_rd (r, a); // handle negative zero
    }
    return r;
}

static __constant__ float cuMinLogDiffFloat = -6.92368989864f;

template <typename Real>
__device__ inline Real log_add(Real x, Real y) {
  Real diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= cuMinLogDiffFloat) {
    Real res;
    res = x + my_log1pf(expf(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

__device__ double atomicLogAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(log_add(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ float atomicLogAdd(float* address, float val) {
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(log_add(val, __int_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

template <typename Real>
__device__ inline void atomic_log_add(Real* address, Real value) {
  atomicLogAdd(address, value);
}


// Similiar to those in chain-kernels.cu, but computed in log-domain.

__global__
static void _cuda_chain_hmm_log_domain_forward(const int *backward_transition_indices,
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

  float this_tot_alpha = -INFINITY;
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

    this_tot_alpha = log_add(this_tot_alpha,
        log_add(this_prev_alpha0 + transition_prob0 + pseudo_loglike0,
          this_prev_alpha1 + transition_prob1 + pseudo_loglike1));
  }
  if (trans_i != trans_end) {
    // mop up the odd transition.
    float transition_prob0 = backward_transition_probs[s * K + trans_i];
    int pdf_id0 = backward_transitions[s * K * 3 + trans_i * 3 + 2],
      prev_hmm_state0 = backward_transitions[s * K * 3 + trans_i * 3];
    float pseudo_loglike0 = probs[s * T * D + (t-1) * D + pdf_id0],
      this_prev_alpha0 = alpha[s * (T+1) * (H+1) + (t-1) * (H+1) + prev_hmm_state0];
    this_tot_alpha = log_add(this_tot_alpha, this_prev_alpha0 + transition_prob0 + pseudo_loglike0);
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
  float arbitrary_scale = -alpha[s * (T+1) * (H+1) + (t-1) * (H+1) + H];
  alpha[s * (T+1) * (H+1) + t * (H+1) + h] = this_tot_alpha + arbitrary_scale;
}


__global__
static void _cuda_chain_hmm_log_domain_backward(const int *forward_transition_indices,
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
    arbitrary_scale = -alpha[s * (T+1) * (H+1)  + t * (H+1) + H];
  float tot_variable_factor = -INFINITY;

  float occupation_factor = this_alpha_prob + arbitrary_scale;
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
    float variable_factor0 = transition_prob0 +
      beta[s * 2 * H + ((t+1) % 2) * H + next_hmm_state0] +
      probs[s * T * D + t * D + pdf_id0];
    float variable_factor1 = transition_prob1 +
      beta[s * 2 * H + ((t+1) % 2) * H + next_hmm_state1] +
      probs[s * T * D + t * D + pdf_id1];
    tot_variable_factor = log_add(tot_variable_factor,
        log_add(variable_factor0, variable_factor1));
    float occupation_prob0 = variable_factor0 + occupation_factor;
    atomic_log_add(log_prob_deriv + s * T * D + t * D + pdf_id0,
                   occupation_prob0);
    float occupation_prob1 = variable_factor1 + occupation_factor;
    atomic_log_add(log_prob_deriv + s * T * D + t * D + pdf_id1,
                   occupation_prob1);
  }
  if (trans_i != trans_end) {
    // mop up the odd transition.
    float transition_prob0 = forward_transition_probs[s * K + trans_i];
    int pdf_id0 = forward_transitions[s * K * 3 + trans_i * 3 + 2],
      next_hmm_state0 = forward_transitions[s * K * 3 + trans_i * 3 + 1];
    float variable_factor0 = transition_prob0 +
      beta[s * 2 * H + ((t+1) % 2) * H + next_hmm_state0] +
      probs[s * T * D + t * D + pdf_id0];
    tot_variable_factor = log_add(tot_variable_factor, variable_factor0);
    float occupation_prob0 = variable_factor0 + occupation_factor;
    atomic_log_add(log_prob_deriv + s * T * D + t * D + pdf_id0,
                   occupation_prob0);
  }
  beta[s * 2 * H + (t%2) * H + h] = tot_variable_factor + arbitrary_scale;
}


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
                                       int num_transitions) {
  _cuda_chain_hmm_log_domain_forward<<<Gr,Bl>>>(backward_transition_indices,
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
                                        int num_transitions) {
  _cuda_chain_hmm_log_domain_backward<<<Gr,Bl>>>(forward_transition_indices,
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
