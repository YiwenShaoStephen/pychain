// chain/chain-denominator.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include "chain-denominator.h"

namespace py = pybind11;

namespace chain {

DenominatorComputation::DenominatorComputation(
    const ChainTrainingOptions &opts,
    const DenominatorGraph &den_graph,
    int32 num_sequences,
    at::Tensor nnet_output):
    opts_(opts),
    den_graph_(den_graph),
    num_sequences_(num_sequences),
    frames_per_sequence_(nnet_output.size(0) / num_sequences_),
    ok_(true) {

  int32 alpha_beta_size = den_graph_.NumStates() * num_sequences_;

#if HAVE_CUDA == 1
  if (nnet_output.is_cuda()) {
    nnet_output_deriv_transposed_ = torch::CUDA(at::kFloat).empty({0, 0});
    alpha_ = torch::CUDA(at::kFloat).empty({0, 0});
    beta_ = torch::CUDA(at::kFloat).empty({0, 0});
    tot_prob_ = torch::CUDA(at::kFloat).empty({0, 0});
    tot_log_prob_ = torch::CUDA(at::kFloat).empty({0, 0});
    log_correction_term_ = torch::CUDA(at::kFloat).empty({0, 0});
    exp_nnet_output_transposed_ = torch::CUDA(at::kFloat).empty({0, 0});
  }
#endif
  nnet_output_deriv_transposed_.resize_({nnet_output.size(1),
    std::min<int32>(nnet_output.size(0),
                    static_cast<int32>(kMaxDerivTimeSteps) *
                    num_sequences_)});
  nnet_output_deriv_transposed_.zero_();
  alpha_.resize_({frames_per_sequence_ + 1,
         alpha_beta_size + num_sequences_});
  beta_.resize_({2, alpha_beta_size + num_sequences_});
  tot_prob_.resize_({num_sequences_});
  tot_log_prob_.resize_({num_sequences_});
  log_correction_term_.resize_({num_sequences_});
  // We don't let leaky_hmm_coefficient be exactly zero (although that would
  // make sense mathematically, corresponding to "turning off" the leaky HMM),
  // because that would lead to underflow and eventually NaN's or inf's
  // appearing in the computation, since we do this computation not in
  // log-space.
  assert(opts_.leaky_hmm_coefficient > 0.0 &&
         opts_.leaky_hmm_coefficient < 1.0);
  // make sure the alpha sums and beta sums are zeroed.
  alpha_.narrow(1, alpha_beta_size, num_sequences_).zero_();
  beta_.narrow(1, alpha_beta_size, num_sequences_).zero_();

  assert(nnet_output.size(0) % num_sequences == 0);
  // the kStrideEqualNumCols argument means we'll allocate a contiguous block of
  // memory for this; it is added to ensure that the same block of memory
  // (cached in the allocator) can be used for xent_output_deriv when allocated
  // from chain-training.cc.
  exp_nnet_output_transposed_.resize_as_(nnet_output.transpose(0, 1));
  exp_nnet_output_transposed_.copy_(nnet_output.transpose(0, 1));
  // We limit the nnet output to the range [-30,30] before doing the exp;
  // this avoids NaNs appearing in the forward-backward computation, which
  // is not done in log space.
  exp_nnet_output_transposed_.clamp_(-30.0, 30.0);
  exp_nnet_output_transposed_.exp_();
  //exp_nnet_output_transposed_.ApplyExpLimited(-30.0, 30.0);
}


void DenominatorComputation::AlphaFirstFrame() {
  at::Tensor first_frame_alpha = alpha_.narrow(0, 0, 1).squeeze();
  // dim == num_hmm_states_ * num_sequences_.
  // 0th row
  // create a 'fake matrix' - view this row as a matrix
  // num_hmm_states_ x num_sequences_.
  // initializer takes [pointer, num-rows, num-cols, stride].
  at::Tensor alpha_mat = first_frame_alpha
    .narrow(0, 0, den_graph_.NumStates() * num_sequences_)
    .view({den_graph_.NumStates(), num_sequences_});

  at::Tensor init_probs = den_graph_.InitialProbs().expand(
      {num_sequences_, den_graph_.NumStates()}).transpose(0, 1);

  alpha_mat.copy_(init_probs);
  // TODO (possible): It would be more efficient here if we implemented a
  // CopyColsFromVec function in class CuMatrix.
  //alpha_mat.SetZero();
  //alpha_mat.AddVecToCols(1.0, den_graph_.InitialProbs(), 0.0);
}


// the alpha computation for some 0 < t <= num_time_steps_.
void DenominatorComputation::AlphaGeneralFrame(int32 t) {
  assert(t > 0 && t <= frames_per_sequence_);
  // Rows t and t-1 of alpha
  at::Tensor this_alpha = alpha_.narrow(0, t, 1).squeeze();
  at::Tensor prev_alpha_dash = alpha_.narrow(0, t - 1, 1).squeeze();

  at::Tensor backward_transition_indices = den_graph_.BackwardTransitionIndices();
  at::Tensor backward_transitions = den_graph_.BackwardTransitions();
  at::Tensor backward_transition_probs = den_graph_.BackwardTransitionProbs();

  int32 num_hmm_states = den_graph_.NumStates(),
      num_sequences = num_sequences_;

  // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
  at::Tensor probs = exp_nnet_output_transposed_.narrow(
      1, (t - 1) * num_sequences_, num_sequences_);

#if HAVE_CUDA == 1
  if (probs.is_cuda()) {
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);

    while (1) {
      if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
        dimGrid.y = 65535;
      // AT_DISPATCH_FLOATING_TYPES(probs.type(), "chain_hmm_forward", ([&] {
      //       cuda_chain_hmm_forward<scalar_t>(dimGrid, dimBlock,
      //                              backward_transition_indices.data<int32>(), 
      //                              backward_transitions.data<int32>(),
      //                              backward_transition_probs.data<scalar_t>(),
      //                              num_sequences, den_graph_.NumStates(),
      //                              probs.data<scalar_t>(), probs.size(1), 
      //                              prev_alpha_dash.data<scalar_t>(), 
      //                              this_alpha.data<scalar_t>());
      //       }));
      cuda_chain_hmm_forward(dimGrid, dimBlock,
			     backward_transition_indices.data<int32>(), 
			     backward_transitions.data<int32>(),
			     backward_transition_probs.data<BaseFloat>(),
			     num_sequences, den_graph_.NumStates(),
			     probs.data<BaseFloat>(), probs.size(1), 
			     prev_alpha_dash.data<BaseFloat>(), 
			     this_alpha.data<BaseFloat>());

      if (dimGrid.y == num_hmm_states) {
        break;  // this is the normal case.
      } else {
        // We reach this code only in the unusual case where num_hmm_states >
        // 65535.  We can compute the alphas for the remaining HMM states by
        // moving some of the array pointers and making the call again.
        //backward_transitions += dimGrid.y;
        backward_transition_indices = backward_transition_indices.narrow(
            0, dimGrid.y, backward_transition_indices.size(1) - dimGrid.y);
        //this_alpha += dimGrid.y * num_sequences;
        this_alpha = this_alpha.narrow(
            0, dimGrid.y * num_sequences, 
            this_alpha.size(0) - dimGrid.y * num_sequences);
        num_hmm_states -= dimGrid.y;
        dimGrid.y = num_hmm_states;
      }
    }
    //CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    auto probs_a = probs.accessor<BaseFloat, 2>();
    auto this_alpha_a = this_alpha.accessor<BaseFloat, 1>();
    auto prev_alpha_dash_a = prev_alpha_dash.accessor<BaseFloat, 1>();
    auto transition_indices_a = backward_transition_indices.accessor<int32, 2>();
    auto transitions_a = backward_transitions.accessor<int32, 2>();
    auto transition_probs_a = backward_transition_probs.accessor<BaseFloat, 1>();

    for (int32 h = 0; h < num_hmm_states; h++) {
      for (int32 s = 0; s < num_sequences; s++) {
        double this_tot_alpha = 0.0;
        for (int32 trans_i = transition_indices_a[h][0];
             trans_i != transition_indices_a[h][1]; trans_i++) {
          BaseFloat transition_prob = transition_probs_a[trans_i];
          int32 pdf_id = transitions_a[trans_i][1],
              prev_hmm_state = transitions_a[trans_i][0];
          BaseFloat prob = probs_a[pdf_id][s],
              this_prev_alpha = prev_alpha_dash_a[prev_hmm_state * num_sequences + s];
          this_tot_alpha += this_prev_alpha * transition_prob * prob;
        }
        // Let arbitrary_scale be the inverse of the alpha-sum value that we
        // store in the same place we'd store the alpha for the state numbered
        // 'num_hmm_states'. We multiply this into all the
        // transition-probabilities from the previous frame to this frame, in
        // both the forward and backward passes, in order to keep the alphas in
        // a good numeric range.  This won't affect the posteriors, but when
        // computing the total likelihood we'll need to compensate for it later
        // on.
        BaseFloat arbitrary_scale =
            1.0 / prev_alpha_dash_a[num_hmm_states * num_sequences + s];
        assert(this_tot_alpha - this_tot_alpha == 0);
        this_alpha_a[h * num_sequences + s] = this_tot_alpha * arbitrary_scale;
      }
    }
  }
}

void DenominatorComputation::AlphaDash(int32 t) {
  // Row t of alpha_ as a vector of size num_hmm_states_ x (num_sequences_ + 1)
  at::Tensor this_alpha = alpha_.narrow(0, t, 1).squeeze();

  // create a 'fake matrix' for the regular alphas- view this row as a matrix.
  // initializer takes [pointer, num-rows, num-cols, stride].
  at::Tensor alpha_mat = this_alpha
    .narrow(0, 0, den_graph_.NumStates() * num_sequences_)
    .view({den_graph_.NumStates(), num_sequences_});

  // Last part of this_alpha stores the sum of alpha over all states
  at::Tensor alpha_sum_vec = this_alpha
    .narrow(0, den_graph_.NumStates() * num_sequences_, num_sequences_);
  // the alpha-dash is the sum of alpha over all states.
 
  alpha_sum_vec.copy_(at::sum(alpha_mat, 0));

  //alpha_mat.AddVecVec(opts_.leaky_hmm_coefficient,
  //                    den_graph_.InitialProbs(),
  //                    alpha_sum_vec);
  alpha_mat.addr_(den_graph_.InitialProbs(), alpha_sum_vec, 1.0, 
                  opts_.leaky_hmm_coefficient);
  // it's now alpha-dash.
}

// compute beta from beta-dash.
void DenominatorComputation::Beta(int32 t) {
  at::Tensor this_beta_dash = beta_.narrow(0, t % 2, 1).squeeze();
  
  int32 this_beta_size = den_graph_.NumStates() * num_sequences_;

  // create a 'fake matrix' for the regular beta-dash (which is
  // the counterpart of alpha-dash)- view this row as a matrix.
  // initializer takes [pointer, num-rows, num-cols, stride].
  at::Tensor beta_dash_mat = this_beta_dash
    .narrow(0, 0, this_beta_size)
    .view({den_graph_.NumStates(), num_sequences_});

  // making the t index implicit, the beta-dash-sum for each sequence is the sum
  // over all states i of beta_i * opts_.leaky_hmm_coefficient * initial_prob_i.
  at::Tensor beta_dash_sum_vec = this_beta_dash
    .narrow(0, this_beta_size, num_sequences_).squeeze();

  beta_dash_sum_vec.addmv_(
      beta_dash_mat.transpose(0, 1), 
      den_graph_.InitialProbs(), 0.0, opts_.leaky_hmm_coefficient);

  // we are computing beta in place.  After the following, beta-dash-mat
  // will contain the actual beta (i.e. the counterpart of alpha),
  // not the beta-dash.
  beta_dash_mat.add_(
      beta_dash_sum_vec
      .expand({den_graph_.NumStates(), num_sequences_}));
}

at::Tensor DenominatorComputation::Forward() {
  AlphaFirstFrame();
  AlphaDash(0);
  for (int32 t = 1; t <= frames_per_sequence_; t++) {
    AlphaGeneralFrame(t);
    AlphaDash(t);
  }
  return ComputeTotLogLike();
}

at::Tensor DenominatorComputation::ComputeTotLogLike() {
  tot_prob_.resize_({num_sequences_});

  int32 alpha_size = den_graph_.NumStates() * num_sequences_;

  // View the last alpha-dash as a matrix of size num-hmm-states by num-sequences.
  at::Tensor last_alpha_dash = alpha_
    .narrow(0, frames_per_sequence_, 1).squeeze()
    .narrow(0, 0, alpha_size)
    .view({den_graph_.NumStates(), num_sequences_});

  // tot_prob_.AddRowSumMat(1.0, last_alpha_dash, 0.0);
  tot_prob_.copy_(at::sum(last_alpha_dash, 0));

  // we should probably add an ApplyLog() function that takes a vector argument.
  tot_log_prob_.resize_as_(tot_prob_);
  tot_log_prob_.copy_(tot_prob_.log());
  at::Tensor tot_log_prob = tot_log_prob_.sum();

  // We now have to add something for the arbitrary scaling factor.  [note: the
  // purpose of the arbitrary scaling factors was to keep things in a good
  // floating-point range]
  // The inverses of all the tot-alpha quantities, for t = 0
  // ... frames_per_sequence_ - 1, were included as the 'arbitrary factors' in
  // the transition-probs, so we need to multiply them all together (not
  // inversed) and add them as a correction term to the total log-likes.
  // These tot-alpha quantities were stored in the same place that we would
  // have stored the HMM-state numbered 'num_hmm_states'.
  // CuSubMatrix<BaseFloat> inv_arbitrary_scales(
  //     alpha_, 0, frames_per_sequence_,
  //     num_sequences_ * num_hmm_states, num_sequences_);
  at::Tensor inv_arbitrary_scales = 
    alpha_.narrow(1, alpha_size, num_sequences_);
  at::Tensor log_inv_arbitrary_scales = at::empty_like(inv_arbitrary_scales);
  log_inv_arbitrary_scales.copy_(inv_arbitrary_scales.log());
  at::Tensor log_inv_arbitrary_scales_product = log_inv_arbitrary_scales.sum();
  return tot_log_prob.add(log_inv_arbitrary_scales_product);
}



bool DenominatorComputation::Backward(at::Tensor nnet_output_deriv) {
  BetaDashLastFrame();
  Beta(frames_per_sequence_);
  for (int32 t = frames_per_sequence_ - 1; t >= 0; t--) {
    BetaDashGeneralFrame(t);
    if (GetVerboseLevel() >= 1 || t == 0)
      BetaGeneralFrameDebug(t);
    Beta(t);
    if (t % kMaxDerivTimeSteps == 0) {
      // commit the derivative stored in nnet_output_deriv_transposed_ by adding
      // its transpose to the appropriate sub-matrix of 'nnet_output_deriv'.
      int32 chunk_frames = std::min<int32>(static_cast<int32>(kMaxDerivTimeSteps),
                                           frames_per_sequence_ - t);

      //CuSubMatrix<BaseFloat> transposed_deriv_part(
      //    nnet_output_deriv_transposed_,
      //    0, num_pdfs,
      //    0, chunk_frames * num_sequences_);
      at::Tensor transposed_deriv_part = 
        nnet_output_deriv_transposed_
        .narrow(1, 0, chunk_frames * num_sequences_);

      //CuSubMatrix<BaseFloat> output_deriv_part(
      //    *nnet_output_deriv,
      //    t * num_sequences_, chunk_frames * num_sequences_,
      //    0, num_pdfs);
      //output_deriv_part.AddMat(deriv_weight, transposed_deriv_part, kTrans);
      at::Tensor output_deriv_part = nnet_output_deriv
        .narrow(0, t * num_sequences_, chunk_frames * num_sequences_);
      output_deriv_part.add_(transposed_deriv_part.transpose(0, 1));
      if (t != 0)
        transposed_deriv_part.zero_();
    }
  }
  return ok_;
}

void DenominatorComputation::BetaDashLastFrame() {
  // sets up the beta-dash quantity on the last frame (frame ==
  // frames_per_sequence_).  Note that the betas we use here contain a
  // 1/(tot-prob) factor in order to simplify the backprop.

  int32 t = frames_per_sequence_;
  at::Tensor last_frame_beta_dash = beta_.narrow(0, t % 2, 1).squeeze();

  // create a 'fake matrix' - view this row as a matrix.
  //CuSubMatrix<BaseFloat> beta_dash_mat(last_frame_beta_dash,
  //                                     den_graph_.NumStates(),
  //                                     num_sequences_,
  //                                     num_sequences_);
  at::Tensor beta_dash_mat = last_frame_beta_dash
    .narrow(0, 0, den_graph_.NumStates() * num_sequences_)
    .view({den_graph_.NumStates(), num_sequences_});

  at::Tensor inv_tot_prob = at::ones_like(tot_prob_);
  inv_tot_prob.div_(tot_prob_);
  // the beta values at the end of the file only vary with the sequence-index,
  // not with the HMM-index.  We treat all states as having a final-prob of one.
  // beta_dash_mat.CopyRowsFromVec(inv_tot_prob);
  beta_dash_mat.copy_(
      inv_tot_prob.expand({den_graph_.NumStates(), num_sequences_}));
}

void DenominatorComputation::BetaDashGeneralFrame(int32 t) {
  assert(t >= 0 && t < frames_per_sequence_);
  // t_wrapped gives us the time-index we use when indexing
  // nnet_output_deriv_transposed_; to save memory we limit the size of the
  // matrix, storing only chunks of frames at a time, and we add it to the
  // non-transposed output whenever we finish a chunk.
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps);
  at::Tensor this_alpha_dash = alpha_.narrow(0, t, 1).squeeze(),
    next_beta = beta_.narrow(0, (t + 1) % 2, 1).squeeze(),
    this_beta_dash = beta_.narrow(0, t % 2, 1).squeeze();

  at::Tensor forward_transition_indices = den_graph_.ForwardTransitionIndices(),
    forward_transitions = den_graph_.ForwardTransitions(),
    forward_transition_probs = den_graph_.ForwardTransitionProbs();

  // 'probs' is the matrix of pseudo-likelihoods for frame t.
  at::Tensor probs = exp_nnet_output_transposed_
    .narrow(1, t * num_sequences_, num_sequences_);
  at::Tensor log_prob_deriv = nnet_output_deriv_transposed_
    .narrow(1, t_wrapped * num_sequences_, num_sequences_);

  int32 num_hmm_states = den_graph_.NumStates(),
      num_sequences = num_sequences_;

#if HAVE_CUDA == 1
  if (probs.is_cuda()) {
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);
    while (1) {
      if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
        dimGrid.y = 65535;
      // AT_DISPATCH_FLOATING_TYPES(probs.type(), "chain_hmm_backward", ([&] {
      //       cuda_chain_hmm_backward<scalar_t>(dimGrid, dimBlock, 
      //                               forward_transition_indices.data<int32>(),
      //                               forward_transitions.data<int32>(),
      //                               forward_transition_probs.data<scalar_t>(),
      //                               num_sequences, num_hmm_states,
      //                               probs.data<scalar_t>(), probs.stride(0),
      //                               this_alpha_dash.data<scalar_t>(), 
      //                               next_beta.data<scalar_t>(),
      //                               this_beta_dash.data<scalar_t>(),
      //                               log_prob_deriv.data<scalar_t>(),
      //                               log_prob_deriv.stride(0));
      //       }));
      cuda_chain_hmm_backward(dimGrid, dimBlock, 
			      forward_transition_indices.data<int32>(),
			      forward_transitions.data<int32>(),
			      forward_transition_probs.data<BaseFloat>(),
			      num_sequences, num_hmm_states,
			      probs.data<BaseFloat>(), probs.stride(0),
			      this_alpha_dash.data<BaseFloat>(), 
			      next_beta.data<BaseFloat>(),
			      this_beta_dash.data<BaseFloat>(),
			      log_prob_deriv.data<BaseFloat>(),
			      log_prob_deriv.stride(0));
      if (dimGrid.y == num_hmm_states) {
        break;  // this is the normal case.
      } else {
        // We reach this code only in the unusual case where num_hmm_states >
        // 65535.  We can compute the betas (and log-prob derivatives) for the
        // remaining HMM states by moving some of the array pointers and making
        // the call again.

        forward_transition_indices = forward_transition_indices.narrow(
            0, dimGrid.y, forward_transition_indices.size(1) - dimGrid.y);
        //this_alpha_dash += dimGrid.y * num_sequences;
        this_alpha_dash = this_alpha_dash.narrow(
            0, dimGrid.y * num_sequences, 
            this_alpha_dash.size(0) - dimGrid.y * num_sequences);
        //this_beta_dash += dimGrid.y * num_sequences;
        this_beta_dash = this_beta_dash.narrow(
            0, dimGrid.y * num_sequences, 
            this_beta_dash.size(0) - dimGrid.y * num_sequences);
        num_hmm_states -= dimGrid.y;
        dimGrid.y = num_hmm_states;
      }
    }
  } else
#endif
  {
    auto probs_a = probs.accessor<BaseFloat, 2>();
    auto log_prob_deriv_a = log_prob_deriv.accessor<BaseFloat, 2>();
    auto this_alpha_dash_a = this_alpha_dash.accessor<BaseFloat, 1>();
    auto this_beta_dash_a = this_beta_dash.accessor<BaseFloat, 1>();
    auto next_beta_a = next_beta.accessor<BaseFloat, 1>();
    auto transition_indices_a = forward_transition_indices.accessor<int32, 2>();
    auto transitions_a = forward_transitions.accessor<int32, 2>();
    auto transition_probs_a = forward_transition_probs.accessor<BaseFloat, 1>();

    for (int32 h = 0; h < num_hmm_states; h++) {
      for (int32 s = 0; s < num_sequences; s++) {
        BaseFloat this_alpha_dash_prob = this_alpha_dash_a[h * num_sequences + s],
            inv_arbitrary_scale =
            this_alpha_dash_a[num_hmm_states * num_sequences + s];
        double tot_variable_factor = 0.0;
        BaseFloat occupation_factor = this_alpha_dash_prob /
            inv_arbitrary_scale;
        
        for (int32 trans_i = transition_indices_a[h][0]; 
             trans_i != transition_indices_a[h][1]; trans_i++) {
          BaseFloat transition_prob = transition_probs_a[trans_i];
          int32 pdf_id = transitions_a[trans_i][1],
              next_hmm_state = transitions_a[trans_i][0];
          BaseFloat variable_factor = transition_prob *
              next_beta_a[next_hmm_state * num_sequences + s] *
              probs_a[pdf_id][s];
          tot_variable_factor += variable_factor;
          BaseFloat occupation_prob = variable_factor * occupation_factor;
          log_prob_deriv_a[pdf_id][s] += occupation_prob;
        }
        this_beta_dash_a[h * num_sequences + s] =
            tot_variable_factor / inv_arbitrary_scale;
      }
    }
  }
}

void DenominatorComputation::BetaGeneralFrameDebug(int32 t) {
  BaseFloat num_hmm_states = den_graph_.NumStates(),
      alpha_beta_size = num_hmm_states * num_sequences_;
  at::Tensor this_alpha_dash = alpha_.narrow(0, t, 1).squeeze()
    .narrow(0, 0, alpha_beta_size);
  at::Tensor this_beta_dash = beta_.narrow(0, t % 2, 1).squeeze()
    .narrow(0, 0, alpha_beta_size);
  
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps);

  at::Tensor this_log_prob_deriv = nnet_output_deriv_transposed_
    .narrow(1, t_wrapped * num_sequences_, num_sequences_);
  
  BaseFloat alpha_beta_product = at::Scalar(at::dot(this_alpha_dash, this_beta_dash)).toFloat(),
      this_log_prob_deriv_sum = at::Scalar(this_log_prob_deriv.sum()).toFloat();

  if (!ApproxEqual(alpha_beta_product, num_sequences_)) {
    std::cerr  << "On time " << t << ", alpha-beta product "
               << alpha_beta_product << " != " << num_sequences_
               << " alpha-dash-sum = " << at::sum(this_alpha_dash)
               << ", beta-dash-sum = " << at::sum(this_beta_dash)
               << std::endl;
    if (fabs(alpha_beta_product - num_sequences_) > 2.0) {
      std::cerr << "Excessive error detected, will abandon this minibatch"
                << std::endl;
      ok_ = false;
    }
  }
  // use higher tolerance, since we are using randomized pruning for the
  // log-prob derivatives.
  if (!ApproxEqual(this_log_prob_deriv_sum,
                   num_sequences_, 0.01)) {
    std::cerr << "On time " << t << ", log-prob-deriv sum "
               << this_log_prob_deriv_sum << " != " << num_sequences_
               << std::endl;
    if (fabs(this_log_prob_deriv_sum - num_sequences_) > 2.0) {
      std::cerr << "Excessive error detected, will abandon this minibatch"
                << std::endl;
      ok_ = false;
    }
  }
}


}  // namespace chain
