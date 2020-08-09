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
    torch::Tensor leaky_probs,
    torch::Tensor initial_probs,
    torch::Tensor final_probs,
    torch::Tensor start_state,
    torch::Tensor exp_nnet_output,
    torch::Tensor batch_sizes,
    torch::Tensor sequence_lengths,
    int num_states, float leaky_hmm_coefficient) {

  cuda_ = exp_nnet_output.type().is_cuda();
  num_sequences_ = (int) exp_nnet_output.size(0);
  num_states_ = num_states;
  num_pdfs_ = (int) exp_nnet_output.size(2);
  num_frames_ = (int) exp_nnet_output.size(1);
  num_transitions_ = (int) forward_transitions.size(1);

  forward_transitions_ = forward_transitions;
  forward_transition_indices_ = forward_transition_indices;
  forward_transition_probs_ = forward_transition_probs;
  backward_transitions_ = backward_transitions;
  backward_transition_indices_ = backward_transition_indices;
  backward_transition_probs_ = backward_transition_probs;
  leaky_probs_ = leaky_probs;
  initial_probs_ = initial_probs;
  final_probs_ = final_probs;
  start_state_ = start_state.to(torch::kLong);

  nnet_output_deriv_ = torch::zeros_like(exp_nnet_output);
  exp_nnet_output_ = exp_nnet_output;
  batch_sizes_ = batch_sizes; // don't need to be put on GPUs
  sequence_lengths_ = sequence_lengths.to(torch::kLong);

  // We don't let leaky_hmm_coefficient be exactly zero (although that would
  // make sense mathematically, corresponding to "turning off" the leaky HMM),
  // because that would lead to underflow and eventually NaN's or inf's
  // appearing in the computation, since we do this computation not in
  // log-space.
  assert(leaky_hmm_coefficient > 0.0 && leaky_hmm_coefficient < 1.0);
  leaky_hmm_coefficient_ = leaky_hmm_coefficient;

  alpha_ = exp_nnet_output.new_zeros({num_sequences_, num_frames_ + 1, num_states_ + 1});
  beta_ = exp_nnet_output.new_zeros({num_sequences_, 2, num_states_});
  tot_prob_ = exp_nnet_output.new_zeros({num_sequences_});
  tot_log_prob_ = exp_nnet_output.new_zeros({num_sequences_});
  ok_ = true;

  if (cuda_) {
    forward_transitions_ = forward_transitions_.cuda();
    forward_transition_indices_ = forward_transition_indices_.cuda();
    forward_transition_probs_ = forward_transition_probs_.cuda();
    backward_transitions_ = backward_transitions_.cuda();
    backward_transition_indices_ = backward_transition_indices_.cuda();
    backward_transition_probs_ = backward_transition_probs_.cuda();
    leaky_probs_ = leaky_probs_.cuda();
    initial_probs_ = initial_probs_.cuda();
    final_probs_ = final_probs_.cuda();
    start_state_ = start_state_.cuda();
    sequence_lengths_ = sequence_lengths_.cuda();
  }
}

void ChainComputation::AlphaFirstFrame() {
  auto alpha_initial_state = alpha_.narrow(1, 0, 1).narrow(2, 0, num_states_).squeeze(1); // B x H
  alpha_initial_state.copy_(initial_probs_);
}

void ChainComputation::AlphaSum(int t) {
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();
  int batch_size;
  if (t == 0) {
    batch_size = num_sequences_;
  } else {
    batch_size = (int) batch_sizes_a[t - 1];
  }
  auto this_alpha = alpha_.narrow(0, 0, batch_size)
    .narrow(1, t, 1).narrow(2, 0, num_states_).squeeze(1); // B x H
  auto this_alpha_tot = alpha_.narrow(0, 0, batch_size)
    .narrow(1, t, 1).narrow(2, num_states_, 1).squeeze(2).squeeze(1); // B
  this_alpha_tot.copy_(this_alpha.sum(1));
}

// the alpha computation for some 0 < t <= num_time_steps_.
void ChainComputation::AlphaGeneralFrame(int t) {
  assert(t > 0 && t <= num_frames_);
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();
  int num_hmm_states = num_states_,
      num_frames = num_frames_,
      num_pdfs = num_pdfs_,
      num_transitions = num_transitions_;
  int num_sequences = (int) batch_sizes_a[t - 1];

  if (cuda_) {
    dim3 dimBlock(std::min<int>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);

    if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
      dimGrid.y = 65535;
    cuda_chain_hmm_forward(dimGrid, dimBlock,
        backward_transition_indices_.data_ptr<int>(),
        backward_transitions_.data_ptr<int>(),
        backward_transition_probs_.data_ptr<float>(),
        exp_nnet_output_.data_ptr<float>(),
        alpha_.data_ptr<float>(),
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
        float arbitrary_scale = 1.0 / prev_alpha_a[s][num_hmm_states];
        assert(this_tot_alpha - this_tot_alpha == 0);
        this_alpha_a[s][h] = this_tot_alpha * arbitrary_scale;
      }
    }
  }
}

void ChainComputation::AlphaDash(int t) {
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();
  int batch_size;
  if (t == 0) {
    batch_size = num_sequences_;
  } else {
    batch_size = (int) batch_sizes_a[t - 1];
  }
  torch::Tensor this_alpha = alpha_.narrow(0, 0, batch_size)
    .narrow(1, t, 1).narrow(2, 0, num_states_).squeeze(1); // B x H
  torch::Tensor this_tot_alpha = alpha_.narrow(0, 0, batch_size)
    .narrow(1, t, 1).narrow(2, num_states_, 1).squeeze(1); // B x 1

  // (B x H) * (B x H) -> B x H
  this_alpha.addcmul_(this_tot_alpha.expand_as(this_alpha),
      leaky_probs_.narrow(0, 0, batch_size), leaky_hmm_coefficient_);
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
  torch::Tensor last_frame_index = sequence_lengths_.unsqueeze(1).unsqueeze(2)
    .expand({-1, -1, alpha_.size(2)}); // B x 1 x (H+1)
  torch::Tensor last_frame_alpha_dash = alpha_.gather(1, last_frame_index)
    .narrow(2, 0, num_states_).squeeze(1); // B x H
  torch::Tensor last_frame_alpha_dash_sum = last_frame_alpha_dash.mul(final_probs_).sum(1); // B

  // Set alpha_tot(T) in each sequence to 0.0 so that its original value will
  // not be added to tot_log_prob_. The original value has already been used in
  // AlphaDash() and from now on it is of no use. B x (T+1) <- B x 1
  alpha_.narrow(2, num_states_, 1).squeeze(2).scatter_(1, sequence_lengths_.unsqueeze(1), 0.0);
  torch::Tensor alpha_frame_tot = alpha_.narrow(1, 0, num_frames_)
    .narrow(2, num_states_, 1).squeeze(2); // B x T
  // padding values (0.0) is unchanged, otherwise apply log
  torch::Tensor alpha_frame_log_tot = torch::where(alpha_frame_tot.eq(0.0),
      alpha_frame_tot.new_zeros({1}), alpha_frame_tot.log()); // B x T

  // as alpha_frame_log_tot is padded with 0.0, the sum below is fine
  tot_log_prob_.copy_(alpha_frame_log_tot.sum(1) + last_frame_alpha_dash_sum.log()); // B
  tot_prob_.copy_(tot_log_prob_.exp()); // B
  return tot_log_prob_.sum();
}

void ChainComputation::BetaDashLastFrame() {
  torch::Tensor last_frame_index = sequence_lengths_.unsqueeze(1).unsqueeze(2)
    .expand({-1, -1, alpha_.size(2)}); // B x 1 x (H+1)
  torch::Tensor last_frame_alpha_dash = alpha_.gather(1, last_frame_index)
    .narrow(2, 0, num_states_).squeeze(1); // B x H
  torch::Tensor last_frame_alpha_dash_sum = last_frame_alpha_dash.mul(final_probs_).sum(1); // B
  torch::Tensor inv_tot_prob = torch::ones_like(last_frame_alpha_dash_sum); // B
  inv_tot_prob.div_(last_frame_alpha_dash_sum); // B
  torch::Tensor last_frame_beta_dash = inv_tot_prob.unsqueeze(1).mul(final_probs_); // B x H

  torch::Tensor last_frame_beta_dash_index = sequence_lengths_.fmod(2)
    .unsqueeze(1).unsqueeze(2).expand({-1, -1, beta_.size(2)}); // B x 1 x H
  beta_.scatter_(1, last_frame_beta_dash_index, last_frame_beta_dash.unsqueeze(1));
}

void ChainComputation::BetaDashGeneralFrame(int t) {
  assert(t >= 0 && t < num_frames_);
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();
  int num_hmm_states = num_states_,
      num_frames = num_frames_,
      num_pdfs = num_pdfs_,
      num_transitions = num_transitions_;
  int num_sequences = (int) batch_sizes_a[t];

  if (cuda_) {
    dim3 dimBlock(std::min<int>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);
    if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
      dimGrid.y = 65535;
    cuda_chain_hmm_backward(dimGrid, dimBlock,
        forward_transition_indices_.data_ptr<int>(),
        forward_transitions_.data_ptr<int>(),
        forward_transition_probs_.data_ptr<float>(),
        exp_nnet_output_.data_ptr<float>(),
        alpha_.data_ptr<float>(),
        beta_.data_ptr<float>(),
        nnet_output_deriv_.data_ptr<float>(),
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

    for (int s = 0; s < num_sequences; s++) {
      float arbitrary_scale = 1.0 / this_alpha_dash_a[s][num_hmm_states];
      for (int h = 0; h < num_hmm_states; h++) {
        float this_alpha_dash_prob = this_alpha_dash_a[s][h];
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
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();
  int batch_size;
  if (t == 0) {
    batch_size = num_sequences_;
  } else {
    batch_size = (int) batch_sizes_a[t - 1];
  }
  torch::Tensor this_beta_dash = beta_.narrow(0, 0, batch_size)
    .narrow(1, t % 2, 1).squeeze(1); // B x H

  // the beta-dash-sum for each sequence is the sum over all states i of
  // beta_i * leaky_hmm_coefficient * leaky_prob_i.
  // sum((B x H) * (B x H)) -> B x 1
  torch::Tensor this_beta_dash_sum = this_beta_dash.mul(leaky_probs_.narrow(0, 0, batch_size)).sum(1, true);
  // (B x H) + (B x 1) -> B x H
  this_beta_dash.add_(this_beta_dash_sum, leaky_hmm_coefficient_);
}

bool ChainComputation::Backward() {
  BetaDashLastFrame();
  Beta(num_frames_);
  for (int t = num_frames_ - 1; t >= 0; t--) {
    BetaDashGeneralFrame(t);
    if (GetVerboseLevel() >= 1 || t == 0)
      BetaGeneralFrameDebug(t);
    Beta(t);
  }
  return ok_;
}


void ChainComputation::BetaGeneralFrameDebug(int t) {
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();
  int batch_size;
  if (t == 0) {
    batch_size = num_sequences_;
  } else {
    batch_size = (int) batch_sizes_a[t - 1];
  }
  int batch_size_next = (int) batch_sizes_a[t];

  int num_hmm_states = num_states_;
  torch::Tensor this_alpha_dash = alpha_.narrow(0, 0, batch_size)
    .narrow(1, t, 1).narrow(2, 0, num_hmm_states).squeeze(1); // B x H
  torch::Tensor this_beta_dash = beta_.narrow(0, 0, batch_size)
    .narrow(1, t % 2, 1).squeeze(1); // B x H

  torch::Tensor this_log_prob_deriv = nnet_output_deriv_.narrow(0, 0, batch_size_next).narrow(1, t, 1);

  float alpha_beta_product = torch::bmm(this_alpha_dash.unsqueeze(1),
      this_beta_dash.unsqueeze(2)).sum().item<float>();
  float this_log_prob_deriv_sum = this_log_prob_deriv.sum().item<float>();

  if (!ApproxEqual(alpha_beta_product, batch_size)) {
    std::cerr << "On time " << t << ", alpha-beta product "
              << alpha_beta_product << " != " << batch_size
              << " alpha-sum = " << this_alpha_dash.sum().item<float>()
              << ", beta-sum = " << this_beta_dash.sum().item<float>()
              << std::endl;
    if (fabs(alpha_beta_product - batch_size) > 0.05 * batch_size) {
      std::cerr << "Excessive error detected, will abandon this minibatch"
                << std::endl;
      ok_ = false;
    }
  }
  // use higher tolerance, since we are using randomized pruning for the
  // log-prob derivatives.
  if (!ApproxEqual(this_log_prob_deriv_sum, batch_size_next, 0.01)) {
    std::cerr << "On time " << t << ", log-prob-deriv sum "
              << this_log_prob_deriv_sum << " != " << batch_size_next
              << std::endl;
    if (fabs(this_log_prob_deriv_sum - batch_size_next) > 0.05 * batch_size_next) {
      std::cerr << "Excessive error detected, will abandon this minibatch"
                << std::endl;
      ok_ = false;
    }
  }
}
