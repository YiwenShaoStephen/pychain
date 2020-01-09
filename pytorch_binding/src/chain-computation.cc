// chain/chain-denominator.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)
//                2019   Yiwen Shao
//                2020   Yiming Wang

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

#include "chain-computation.h"
#include "chain-kernels-ansi.h"
#include "base.h"

ChainComputation::ChainComputation(
    torch::Tensor forward_transitions,
    torch::Tensor forward_transition_indices,
    torch::Tensor forward_transition_probs,
    torch::Tensor backward_transitions,
    torch::Tensor backward_transition_indices,
    torch::Tensor backward_transition_probs,
    torch::Tensor initial_probs,
    torch::Tensor final_probs,
    torch::Tensor exp_nnet_output,
    int num_states, float leaky_hmm_coefficient) {
  
  
  cuda_ = exp_nnet_output.type().is_cuda();
  num_sequences_ = exp_nnet_output.size(0);
  num_states_ = num_states;
  num_pdfs_ = exp_nnet_output.size(2);
  num_frames_ = exp_nnet_output.size(1);
  num_transitions_ = forward_transitions.size(1);

  forward_transitions_ = forward_transitions;
  forward_transition_indices_ = forward_transition_indices;
  forward_transition_probs_ = forward_transition_probs;
  backward_transitions_ = backward_transitions;
  backward_transition_indices_ = backward_transition_indices;
  backward_transition_probs_ = backward_transition_probs;
  initial_probs_ = initial_probs;
  final_probs_ = final_probs;

  nnet_output_deriv_ = torch::zeros_like(exp_nnet_output);
  exp_nnet_output_ = exp_nnet_output;

  // We don't let leaky_hmm_coefficient be exactly zero (although that would
  // make sense mathematically, corresponding to "turning off" the leaky HMM),
  // because that would lead to underflow and eventually NaN's or inf's
  // appearing in the computation, since we do this computation not in
  // log-space.
  assert(leaky_hmm_coefficient > 0.0 && leaky_hmm_coefficient < 1.0);
  leaky_hmm_coefficient_ = leaky_hmm_coefficient;

  alpha_ = torch::zeros({num_sequences_, num_frames_ + 1, num_states_ + 1}, torch::kFloat);
  beta_ = torch::zeros({num_sequences_, 2, num_states_}, torch::kFloat);
  tot_prob_ = torch::zeros({num_sequences_}, torch::kFloat);
  tot_log_prob_ = torch::zeros({num_sequences_}, torch::kFloat);
  
  if(cuda_) {
    forward_transitions_ = forward_transitions_.cuda();
    forward_transition_indices_ = forward_transition_indices_.cuda();
    forward_transition_probs_ = forward_transition_probs_.cuda();
    backward_transitions_ = backward_transitions_.cuda();
    backward_transition_indices_ = backward_transition_indices_.cuda();
    backward_transition_probs_ = backward_transition_probs_.cuda();
    initial_probs_ = initial_probs_.cuda();
    final_probs_ = final_probs_.cuda();
    alpha_ = alpha_.cuda();
    beta_ = beta_.cuda();
    tot_prob_ = tot_prob_.cuda();
    tot_log_prob_ = tot_log_prob_.cuda();
  }
  final_probs_all_ones_ = final_probs_.eq(1.0).all().item<bool>();
}

void ChainComputation::AlphaFirstFrame() {
  auto first_frame_alpha = alpha_.narrow(1, 0, 1).narrow(2, 0, num_states_);
  
  auto init_probs_ex = initial_probs_.expand_as(first_frame_alpha);

  first_frame_alpha.copy_(init_probs_ex);
}

void ChainComputation::AlphaSum(int t) {
  auto this_alpha = alpha_.narrow(1, t, 1).narrow(2, 0, num_states_).squeeze(1); // B x H
  auto this_alpha_tot = alpha_.narrow(1, t, 1).narrow(2, num_states_, 1).squeeze(2).squeeze(1); // B
  this_alpha_tot.copy_(this_alpha.sum(1));
  if (!final_probs_all_ones_ && t == num_frames_) // add final-probs for the last frame
    this_alpha_tot.copy_(this_alpha.mul(final_probs_).sum(1));
  else
    this_alpha_tot.copy_(this_alpha.sum(1));
}

// the alpha computation for some 0 < t <= num_time_steps_.
void ChainComputation::AlphaGeneralFrame(int t) {
  assert(t > 0 && t <= num_frames_);
  int num_hmm_states = num_states_,
    num_sequences = num_sequences_,
    num_frames = num_frames_,
    num_pdfs = num_pdfs_,
    num_transitions = num_transitions_;

  if (cuda_) {
    dim3 dimBlock(std::min<int>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);

    if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
      dimGrid.y = 65535;
    cuda_chain_hmm_forward(dimGrid, dimBlock,
			   backward_transition_indices_.data<int>(), 
			   backward_transitions_.data<int>(),
			   backward_transition_probs_.data<float>(),
			   exp_nnet_output_.data<float>(),
			   alpha_.data<float>(),
			   t, num_sequences, num_frames,
			   num_hmm_states, num_pdfs, num_transitions);
  } else
  {
    // Rows t and t-1 of alpha
    torch::Tensor this_alpha = alpha_.narrow(1, t, 1).squeeze(1);
    torch::Tensor prev_alpha = alpha_.narrow(1, t - 1, 1).squeeze(1);
    // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
    torch::Tensor probs = exp_nnet_output_.narrow(1, t - 1, 1).squeeze(1);
    auto probs_a = probs.accessor<float, 2>();
    auto this_alpha_a = this_alpha.accessor<float, 2>();
    auto prev_alpha_a = prev_alpha.accessor<float, 2>();
    auto transition_indices_a = backward_transition_indices_.accessor<int, 3>();
    auto transitions_a = backward_transitions_.accessor<int, 3>();
    auto transition_probs_a = backward_transition_probs_.accessor<float, 2>();

    for (int s = 0; s < num_sequences; s++) {
      for (int h = 0; h < num_hmm_states; h++) {
        float this_tot_alpha = 0.0;
        for (int trans_i = transition_indices_a[s][h][0];
            trans_i != transition_indices_a[s][h][1]; trans_i++) {
          float transition_prob = transition_probs_a[s][trans_i];
          int pdf_id = transitions_a[s][trans_i][2],
              prev_hmm_state = transitions_a[s][trans_i][0];
          float prob = probs_a[s][pdf_id],
              this_prev_alpha = prev_alpha_a[s][prev_hmm_state];
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
        float arbitrary_scale =
	  1.0 / prev_alpha_a[s][num_hmm_states];
        assert(this_tot_alpha - this_tot_alpha == 0);
        this_alpha_a[s][h] = this_tot_alpha * arbitrary_scale;
      }
    }
  }
}

void ChainComputation::AlphaDash(int t) {
  torch::Tensor this_alpha = alpha_.narrow(1, t, 1).narrow(2, 0, num_states_).squeeze(1); // B x H
  torch::Tensor this_tot_alpha = alpha_.narrow(1, t, 1).narrow(2, num_states_, 1).squeeze(1); // B x 1

  // (B x 1) * (1 x H) -> B x H
  this_alpha.addmm_(this_tot_alpha, initial_probs_.unsqueeze(0), 1.0, leaky_hmm_coefficient_);
}

torch::Tensor ChainComputation::Forward() {
  AlphaFirstFrame();
  AlphaSum(0);
  AlphaDash(0);
  for (int t = 1; t <= num_frames_; t++) {
    AlphaGeneralFrame(t);
    AlphaSum(t);
    AlphaDash(t);
  }
  auto obj = ComputeTotLogLike();
  return obj;
}

torch::Tensor ChainComputation::ComputeTotLogLike() {

  torch::Tensor alpha_frame_log_tot = alpha_.narrow(2, num_states_, 1).squeeze(2).log();
  
  tot_log_prob_.copy_(torch::sum(alpha_frame_log_tot, 1));
  tot_prob_.copy_(tot_log_prob_.exp());

  return tot_log_prob_.sum();
}

void ChainComputation::BetaDashLastFrame() {
  // sets up the beta-dash quantity on the last frame (frame ==
  // num_frames_).  Note that the betas we use here contain a
  // 1/(tot-prob) factor in order to simplify the backprop.
  // the beta values at the end of the file only vary with the sequence-index,
  // not with the HMM-index.  We treat all states as having a final-prob of one
  // for denominator.

  torch::Tensor last_frame_beta_dash = beta_.narrow(1, num_frames_ % 2, 1).squeeze(1); // B x H

  torch::Tensor last_frame_alpha_dash_sum = alpha_.narrow(1, num_frames_, 1)
    .narrow(2, num_states_, 1).squeeze(2).squeeze(1); // B
  torch::Tensor inv_tot_prob = torch::ones_like(last_frame_alpha_dash_sum);
  inv_tot_prob.div_(last_frame_alpha_dash_sum);

  if (final_probs_all_ones_)
    last_frame_beta_dash.copy_(inv_tot_prob.unsqueeze(1).expand_as(last_frame_beta_dash));
  else
    last_frame_beta_dash.copy_(
        inv_tot_prob.unsqueeze(1).expand_as(last_frame_beta_dash).mul(final_probs_));
}

void ChainComputation::BetaDashGeneralFrame(int t) {
  assert(t >= 0 && t < num_frames_);
  int num_hmm_states = num_states_,
    num_sequences = num_sequences_,
    num_frames = num_frames_,
    num_pdfs = num_pdfs_,
    num_transitions = num_transitions_;
  
  if (cuda_) {
    dim3 dimBlock(std::min<int>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);
    if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
      dimGrid.y = 65535;
    cuda_chain_hmm_backward(dimGrid, dimBlock, 
			    forward_transition_indices_.data<int>(),
			    forward_transitions_.data<int>(),
			    forward_transition_probs_.data<float>(),
			    exp_nnet_output_.data<float>(),
			    alpha_.data<float>(),
			    beta_.data<float>(),
			    nnet_output_deriv_.data<float>(),
			    t, num_sequences, num_frames,
			    num_hmm_states, num_pdfs, num_transitions);
  } else
  {
    torch::Tensor this_alpha_dash = alpha_.narrow(1, t, 1).squeeze(1),
      next_beta = beta_.narrow(1, (t + 1) % 2, 1).squeeze(1),
      this_beta_dash = beta_.narrow(1, t % 2, 1).squeeze(1);
    // 'probs' is the matrix of pseudo-likelihoods for frame t.
    torch::Tensor probs = exp_nnet_output_.narrow(1, t, 1).squeeze(1);
    torch::Tensor log_prob_deriv = nnet_output_deriv_.narrow(1, t, 1).squeeze(1);

    auto probs_a = probs.accessor<float, 2>();
    auto log_prob_deriv_a = log_prob_deriv.accessor<float, 2>();
    auto this_alpha_dash_a = this_alpha_dash.accessor<float, 2>();
    auto this_beta_dash_a = this_beta_dash.accessor<float, 2>();
    auto next_beta_a = next_beta.accessor<float, 2>();
    auto transition_indices_a = forward_transition_indices_.accessor<int, 3>();
    auto transitions_a = forward_transitions_.accessor<int, 3>();
    auto transition_probs_a = forward_transition_probs_.accessor<float, 2>();

    for (int h = 0; h < num_hmm_states; h++) {
      for (int s = 0; s < num_sequences; s++) {
        float this_alpha_dash_prob = this_alpha_dash_a[s][h],
            arbitrary_scale = 1.0 / this_alpha_dash_a[s][num_hmm_states];
        float tot_variable_factor = 0.0;
        float occupation_factor = this_alpha_dash_prob * arbitrary_scale;
        for (int trans_i = transition_indices_a[s][h][0]; 
            trans_i != transition_indices_a[s][h][1]; trans_i++) {
          float transition_prob = transition_probs_a[s][trans_i];
          int pdf_id = transitions_a[s][trans_i][2],
              next_hmm_state = transitions_a[s][trans_i][1];
          float variable_factor = transition_prob *
              next_beta_a[s][next_hmm_state] *
              probs_a[s][pdf_id];
          tot_variable_factor += variable_factor;
          float occupation_prob = variable_factor * occupation_factor;
          log_prob_deriv_a[s][pdf_id] += occupation_prob;
        }
        this_beta_dash_a[s][h] = tot_variable_factor * arbitrary_scale;
      }
    }
  }
}

void ChainComputation::Beta(int t) {
  torch::Tensor this_beta_dash = beta_.narrow(1, t % 2, 1).squeeze(1); // B x H

  // the beta-dash-sum for each sequence is the sum over all states i of
  // beta_i * leaky_hmm_coefficient * initial_prob_i.
  // (B x H) * (H x 1) -> B x 1
  torch::Tensor this_beta_dash_sum = torch::mm(this_beta_dash, initial_probs_.unsqueeze(1));
  // (B x H) + (B x 1) -> B x H
  this_beta_dash.add_(this_beta_dash_sum, leaky_hmm_coefficient_);
}

bool ChainComputation::Backward() {
  BetaDashLastFrame();
  Beta(num_frames_);
  for (int t = num_frames_ - 1; t >= 0; t--) {
    BetaDashGeneralFrame(t);
    if (GetVerboseLevel() >= 1 || t ==0)
      BetaGeneralFrameDebug(t);
    Beta(t);
  }
  return ok_;
}


void ChainComputation::BetaGeneralFrameDebug(int t) {
  int num_hmm_states = num_states_;
  torch::Tensor this_alpha_dash = alpha_.narrow(1, t, 1).narrow(2, 0, num_hmm_states).squeeze(1),
    this_beta_dash = beta_.narrow(1, t % 2, 1).squeeze(1);

  torch::Tensor this_log_prob_deriv = nnet_output_deriv_.narrow(1, t, 1);

  float alpha_beta_product = torch::bmm(this_alpha_dash.unsqueeze(1), this_beta_dash.unsqueeze(2)).sum().cpu().data<float>()[0],
    this_log_prob_deriv_sum = this_log_prob_deriv.sum().cpu().data<float>()[0];

  if (!ApproxEqual(alpha_beta_product, num_sequences_)) {
    std::cerr  << "On time " << t << ", alpha-beta product "
               << alpha_beta_product << " != " << num_sequences_
               << " alpha-sum = " << torch::sum(this_alpha_dash)
               << ", beta-sum = " << torch::sum(this_beta_dash)
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
