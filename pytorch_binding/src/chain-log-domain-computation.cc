// chain-log-domain-computation.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)
//                2019   Yiwen Shao
//                2020   Yiming Wang
//                2020   Facebook Inc.  (author: Vimal Manohar)

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

#include "chain-log-domain-computation.h"
#include "chain-kernels-ansi.h"
#include "base.h"

ChainLogDomainComputation::ChainLogDomainComputation(
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

  cuda_ = nnet_output.type().is_cuda();
  num_sequences_ = (int) nnet_output.size(0);
  num_states_ = num_states;
  num_pdfs_ = (int) nnet_output.size(2);
  num_frames_ = (int) nnet_output.size(1);
  num_transitions_ = (int) forward_transitions.size(1);

  forward_transitions_ = forward_transitions;
  forward_transition_indices_ = forward_transition_indices;
  forward_transition_probs_ = forward_transition_probs;
  backward_transitions_ = backward_transitions;
  backward_transition_indices_ = backward_transition_indices;
  backward_transition_probs_ = backward_transition_probs;
  initial_probs_ = initial_probs;
  final_probs_ = final_probs;
  start_state_ = start_state.to(torch::kLong);

  nnet_output_deriv_ = torch::full_like(nnet_output, -std::numeric_limits<float>::infinity());
  nnet_output_ = nnet_output;
  batch_sizes_ = batch_sizes; // don't need to be put on GPUs
  sequence_lengths_ = sequence_lengths.to(torch::kLong);

  alpha_ = nnet_output.new_full({num_sequences_, num_frames_ + 1, num_states_ + 1},
      -std::numeric_limits<float>::infinity());
  beta_ = nnet_output.new_full({num_sequences_, 2, num_states_},
      -std::numeric_limits<float>::infinity());
  tot_log_prob_ = nnet_output.new_full({num_sequences_},
      -std::numeric_limits<float>::infinity());
  ok_ = true;

  if (cuda_) {
    forward_transitions_ = forward_transitions_.cuda();
    forward_transition_indices_ = forward_transition_indices_.cuda();
    forward_transition_probs_ = forward_transition_probs_.cuda();
    backward_transitions_ = backward_transitions_.cuda();
    backward_transition_indices_ = backward_transition_indices_.cuda();
    backward_transition_probs_ = backward_transition_probs_.cuda();
    initial_probs_ = initial_probs_.cuda();
    final_probs_ = final_probs_.cuda();
    start_state_ = start_state_.cuda();
    sequence_lengths_ = sequence_lengths_.cuda();
  }
}

void ChainLogDomainComputation::AlphaFirstFrame() {
  auto alpha_initial_state = alpha_.narrow(1, 0, 1).narrow(2, 0, num_states_).squeeze(1); // B x H
  alpha_initial_state.copy_(initial_probs_);  // usually would be 0 at start_state and -inf otherwise

  // For alpha-sum
  alpha_.narrow(1, 0, 1).narrow(2, num_states_, 1).fill_(0.0);
}

// the alpha computation for some 0 < t <= num_time_steps_.
void ChainLogDomainComputation::AlphaGeneralFrame(int t) {
  assert(t > 0 && t <= num_frames_);
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();
  int num_hmm_states = num_states_,
      num_frames = num_frames_,
      num_pdfs = num_pdfs_,
      num_transitions = num_transitions_;
  int num_sequences = (int) batch_sizes_a[t - 1];

  torch::Tensor this_alpha = alpha_.narrow(0, 0, num_sequences)
    .narrow(1, t, 1).narrow(2, 0, num_hmm_states).squeeze(1); // B x H

  if (cuda_) {
    dim3 dimBlock(std::min<int>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);

    if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
      dimGrid.y = 65535;
    cuda_chain_hmm_log_domain_forward(dimGrid, dimBlock,
        backward_transition_indices_.data_ptr<int>(),
        backward_transitions_.data_ptr<int>(),
        backward_transition_probs_.data_ptr<float>(),
        nnet_output_.data_ptr<float>(),
        alpha_.data_ptr<float>(),
        t, num_sequences, num_frames,
        num_hmm_states, num_pdfs, num_transitions);
  } else
  {
    torch::Tensor prev_alpha = alpha_.narrow(0, 0, num_sequences)
      .narrow(1, t - 1, 1).narrow(2, 0, num_hmm_states).squeeze(1); // B x H
    // 'probs' is the matrix of pseudo-log-likelihoods for frame t - 1.
    torch::Tensor probs = nnet_output_.narrow(1, t - 1, 1).squeeze(1);
    auto probs_a = probs.accessor<float, 2>();
    auto this_alpha_a = this_alpha.accessor<float, 2>(); 
    auto prev_alpha_a = prev_alpha.accessor<float, 2>();
    auto transition_indices_a = backward_transition_indices_.accessor<int, 3>();
    auto transitions_a = backward_transitions_.accessor<int, 3>();
    auto transition_probs_a = backward_transition_probs_.accessor<float, 2>();

    for (int s = 0; s < num_sequences; s++) {
      for (int h = 0; h < num_hmm_states; h++) {
        for (int trans_i = transition_indices_a[s][h][0];
            trans_i != transition_indices_a[s][h][1]; trans_i++) {
          float transition_prob = transition_probs_a[s][trans_i];
          int pdf_id = transitions_a[s][trans_i][2],
              prev_hmm_state = transitions_a[s][trans_i][0];
          float prob = probs_a[s][pdf_id],
              this_prev_alpha = prev_alpha_a[s][prev_hmm_state];
          this_alpha_a[s][h] = LogAdd(this_alpha_a[s][h],
                                      this_prev_alpha + transition_prob + prob);
        }
        assert(this_alpha_a[s][h] - this_alpha_a[s][h] == 0);
      }
    }

    // Add arbitrary scales i.e. Log(1 / prev alpha-sum)
    // to keep everything small and in good range
    torch::Tensor prev_alpha_sums = alpha_.narrow(0, 0, num_sequences)
      .narrow(1, t - 1, 1).narrow(2, num_hmm_states, 1).squeeze(1); // B x 1
    this_alpha.add_(-prev_alpha_sums.expand_as(this_alpha));  // B x H
  }

  // Compute alpha-sum to use as arbitrary scale
  torch::Tensor this_alpha_sums = alpha_.narrow(0, 0, num_sequences)
    .narrow(1, t, 1).narrow(2, num_hmm_states, 1).squeeze(2).squeeze(1); // B
  this_alpha_sums.copy_(this_alpha.logsumexp(1)); // B
}

torch::Tensor ChainLogDomainComputation::Forward() {
  AlphaFirstFrame();
  for (int t = 1; t <= num_frames_; t++) {
    AlphaGeneralFrame(t);
  }
  auto obj = ComputeTotLogLike();
  return obj;
}

torch::Tensor ChainLogDomainComputation::ComputeTotLogLike() {
  torch::Tensor last_frame_index = sequence_lengths_.unsqueeze(1).unsqueeze(2)
    .expand({-1, -1, alpha_.size(2)}); // B x 1 x (H+1)
  torch::Tensor last_frame_alpha = alpha_.gather(1, last_frame_index)
    .narrow(2, 0, num_states_).squeeze(1); // B x H
  torch::Tensor last_frame_alpha_sum = last_frame_alpha.add(final_probs_).logsumexp(1); // B

  // Set alpha_tot(T) in each sequence to 0.0 so that its original value will
  // not be added to tot_log_prob_. B x (T+1) <- B x 1
  alpha_.narrow(2, num_states_, 1).squeeze(2).scatter_(1, sequence_lengths_.unsqueeze(1), 0.0);
  torch::Tensor alpha_frame_tot = alpha_.narrow(1, 0, num_frames_)
    .narrow(2, num_states_, 1).squeeze(2); // B x T
  // replace "-inf" with 0 in alpha_frame_tot, so that "-inf" will not been
  // added to tot_log_prob_
  torch::Tensor alpha_frame_log_tot = torch::where(
      alpha_frame_tot.ne(-std::numeric_limits<float>::infinity()),
      alpha_frame_tot, alpha_frame_tot.new_zeros({1}));

  tot_log_prob_.copy_(alpha_frame_log_tot.sum(1) + last_frame_alpha_sum); // B
  return tot_log_prob_.sum();
}

void ChainLogDomainComputation::BetaLastFrame() {
  torch::Tensor last_frame_index = sequence_lengths_.unsqueeze(1).unsqueeze(2)
    .expand({-1, -1, alpha_.size(2)}); // B x 1 x (H+1)
  torch::Tensor last_frame_alpha = alpha_.gather(1, last_frame_index)
    .narrow(2, 0, num_states_).squeeze(1); // B x H
  torch::Tensor last_frame_alpha_sum = last_frame_alpha.add(final_probs_).logsumexp(1); // B
  torch::Tensor last_frame_beta = final_probs_.add(-last_frame_alpha_sum.unsqueeze(1)); // B x H
  torch::Tensor last_frame_beta_index = sequence_lengths_.fmod(2)
    .unsqueeze(1).unsqueeze(2).expand({-1, -1, beta_.size(2)}); // B x 1 x H
  beta_.scatter_(1, last_frame_beta_index, last_frame_beta.unsqueeze(1));
}

void ChainLogDomainComputation::BetaGeneralFrame(int t) {
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
    cuda_chain_hmm_log_domain_backward(dimGrid, dimBlock,
        forward_transition_indices_.data_ptr<int>(),
        forward_transitions_.data_ptr<int>(),
        forward_transition_probs_.data_ptr<float>(),
        nnet_output_.data_ptr<float>(),
        alpha_.data_ptr<float>(),
        beta_.data_ptr<float>(),
        nnet_output_deriv_.data_ptr<float>(),
        t, num_sequences, num_frames,
        num_hmm_states, num_pdfs, num_transitions);
  } else
  {
    torch::Tensor this_alpha = alpha_.narrow(0, 0, num_sequences)
      .narrow(1, t, 1).narrow(2, 0, num_hmm_states).squeeze(1); // B x H
    torch::Tensor next_beta = beta_.narrow(0, 0, num_sequences)
      .narrow(1, (t + 1) % 2, 1).squeeze(1); // B x H
    torch::Tensor this_beta = beta_.narrow(0, 0, num_sequences)
      .narrow(1, t % 2, 1).squeeze(1); // B x H
    // 'probs' is the matrix of pseudo-log-likelihoods for frame t.
    torch::Tensor probs = nnet_output_.narrow(1, t, 1).squeeze(1); // B x V
    torch::Tensor log_prob_deriv = nnet_output_deriv_.narrow(1, t, 1).squeeze(1); // B x V

    auto probs_a = probs.accessor<float, 2>();
    auto log_prob_deriv_a = log_prob_deriv.accessor<float, 2>();
    auto this_alpha_a = this_alpha.accessor<float, 2>();
    auto this_beta_a = this_beta.accessor<float, 2>();
    auto next_beta_a = next_beta.accessor<float, 2>();
    auto transition_indices_a = forward_transition_indices_.accessor<int, 3>();
    auto transitions_a = forward_transitions_.accessor<int, 3>();
    auto transition_probs_a = forward_transition_probs_.accessor<float, 2>();

    for (int s = 0; s < num_sequences; s++) {
      float inv_arbitrary_scale = this_alpha_a[s][num_hmm_states];
      for (int h = 0; h < num_hmm_states; h++) {
        float this_alpha_prob = this_alpha_a[s][h];
        float tot_variable_factor = -std::numeric_limits<float>::infinity();
        for (int trans_i = transition_indices_a[s][h][0];
            trans_i != transition_indices_a[s][h][1]; trans_i++) {
          float transition_prob = transition_probs_a[s][trans_i];
          int pdf_id = transitions_a[s][trans_i][2],
              next_hmm_state = transitions_a[s][trans_i][1];
          float variable_factor = transition_prob +
              next_beta_a[s][next_hmm_state] +
              probs_a[s][pdf_id] - inv_arbitrary_scale;
          tot_variable_factor = LogAdd(tot_variable_factor, variable_factor);
          float occupation_prob = variable_factor + this_alpha_prob;
          log_prob_deriv_a[s][pdf_id] = LogAdd(log_prob_deriv_a[s][pdf_id],
                                               occupation_prob);
        }
        this_beta_a[s][h] = tot_variable_factor;
      }
    }
  }
}

bool ChainLogDomainComputation::Backward() {
  BetaLastFrame();
  for (int t = num_frames_ - 1; t >= 0; t--) {
    BetaGeneralFrame(t);
    if (GetVerboseLevel() >= 1 || t == 0)
      BetaGeneralFrameDebug(t);
  }
  return ok_;
}

void ChainLogDomainComputation::BetaGeneralFrameDebug(int t) {
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();
  int batch_size_next = (int) batch_sizes_a[t];

  torch::Tensor this_log_prob_deriv = nnet_output_deriv_.narrow(0, 0, batch_size_next).narrow(1, t, 1);

  float this_log_prob_deriv_sum = this_log_prob_deriv.exp().sum().item<float>();

  // use higher tolerance, since we are using randomized pruning for the
  // log-prob derivatives.
  if (!ApproxEqual(this_log_prob_deriv_sum, batch_size_next, 0.01)) {
    std::cerr << "On time " << t << ", log-prob-deriv sum "
              << this_log_prob_deriv_sum << " != " << batch_size_next
              << std::endl;
    if (fabs(this_log_prob_deriv_sum - batch_size_next) > 0.05 * batch_size_next ||
        this_log_prob_deriv_sum - this_log_prob_deriv_sum != 0) {
      std::cerr << "Excessive error detected, will abandon this minibatch"
                << std::endl;
      ok_ = false;
    }
  }
}
