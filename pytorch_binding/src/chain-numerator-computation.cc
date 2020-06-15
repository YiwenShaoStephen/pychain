// chain-numerator-computation.cc

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

#include "chain-numerator-computation.h"
#include "base.h"

ChainNumeratorComputation::ChainNumeratorComputation(
    torch::Tensor forward_transitions,
    torch::Tensor forward_transition_indices,
    torch::Tensor forward_transition_probs,
    torch::Tensor backward_transitions,
    torch::Tensor backward_transition_indices,
    torch::Tensor backward_transition_probs,
    torch::Tensor final_probs,
    torch::Tensor start_state,
    torch::Tensor nnet_output,
    torch::Tensor batch_sizes,
    torch::Tensor sequence_lengths,
    int num_states) {

  num_sequences_ = nnet_output.size(0);
  num_states_ = num_states;
  num_pdfs_ = nnet_output.size(2);
  num_frames_ = nnet_output.size(1);
  num_transitions_ = forward_transitions.size(1);

  forward_transitions_ = forward_transitions;
  forward_transition_indices_ = forward_transition_indices;
  forward_transition_probs_ = forward_transition_probs;
  backward_transitions_ = backward_transitions;
  backward_transition_indices_ = backward_transition_indices;
  backward_transition_probs_ = backward_transition_probs;
  final_probs_ = final_probs;
  start_state_ = start_state;

  nnet_output_deriv_ = torch::empty_like(nnet_output).fill_(
    -std::numeric_limits<float>::infinity());
  nnet_output_ = nnet_output;
  batch_sizes_ = batch_sizes;
  sequence_lengths_ = sequence_lengths;

  alpha_ = torch::empty({num_sequences_, num_frames_ + 1, num_states_ + 1}, torch::kFloat).fill_(
    -std::numeric_limits<float>::infinity());
  beta_ = torch::empty({num_sequences_, 2, num_states_}, torch::kFloat).fill_(
    -std::numeric_limits<float>::infinity());
  tot_log_prob_ = torch::empty({num_sequences_}, torch::kFloat).fill_(
    -std::numeric_limits<float>::infinity());
}

void ChainNumeratorComputation::AlphaFirstFrame() {
  auto alpha_initial_state = alpha_.narrow(1, 0, 1).narrow(2, 0, num_states_).squeeze(1); // B x H
  alpha_initial_state.scatter_(1, start_state_.unsqueeze(1), 0.0);   // Set initial log-prob to Log(1.0)

  // For alpha-sum
  alpha_.narrow(1, 0, 1).narrow(2, num_states_, 1).fill_(0.0);

  if (GetVerboseLevel() >= 3)
    std::cerr << "alpha initial = " << alpha_initial_state;
}

// the alpha computation for some 0 < t <= num_time_steps_.
void ChainNumeratorComputation::AlphaRemainingFrames() {
  int num_hmm_states = num_states_,
    num_frames = num_frames_,
    num_pdfs = num_pdfs_,
    num_transitions = num_transitions_;
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();

  auto transition_indices_a = backward_transition_indices_.accessor<int, 3>();
  auto transitions_a = backward_transitions_.accessor<int, 3>();
  auto transition_probs_a = backward_transition_probs_.accessor<float, 2>();
  for (int t = 1; t <= num_frames_; t++) {
    long num_sequences = batch_sizes_a[t - 1];

    torch::Tensor alpha_t = alpha_.narrow(0, 0, num_sequences)
                                .narrow(1, t, 1)
                                .narrow(2, 0, num_hmm_states)
                                .squeeze(1); // B x H
    torch::Tensor alpha_tm1 = alpha_.narrow(0, 0, num_sequences)
                                  .narrow(1, t - 1, 1)
                                  .narrow(2, 0, num_hmm_states)
                                  .squeeze(1); // B x H

    // 'probs' is the matrix of pseudo-log-likelihoods for frame t - 1.
    torch::Tensor probs = nnet_output_.narrow(1, t - 1, 1).squeeze(1);  // B x V

    auto probs_a = probs.accessor<float, 2>();
    auto alpha_t_a = alpha_t.accessor<float, 2>();
    auto alpha_tm1_a = alpha_tm1.accessor<float, 2>();

    for (int s = 0; s < num_sequences; s++) {
      for (int h = 0; h < num_hmm_states; h++) {
        float this_tot_alpha = 0.0;
        for (int trans_i = transition_indices_a[s][h][0];
            trans_i != transition_indices_a[s][h][1]; trans_i++) {
          float transition_prob = transition_probs_a[s][trans_i];
          int pdf_id = transitions_a[s][trans_i][2],
              prev_hmm_state = transitions_a[s][trans_i][0];
          float prob = probs_a[s][pdf_id],
              this_prev_alpha = alpha_tm1_a[s][prev_hmm_state];
          alpha_t_a[s][h] = LogAdd(alpha_t_a[s][h],
            this_prev_alpha + transition_prob + prob);
        }
      }
    }

    // Add arbitrary scales i.e. Log(1 / prev alpha-sum)
    // to keep everything small and in good range
    torch::Tensor prev_alpha_sums = alpha_.narrow(0, 0, num_sequences)
                                        .narrow(1, t - 1, 1)
                                        .narrow(2, num_hmm_states, 1)
                                        .squeeze(1); // B x 1
    alpha_t.add_(-prev_alpha_sums.expand_as(alpha_t));  // B x H

    // Compute alpha-sum to use as arbitrary scale
    torch::Tensor this_alpha_sums = alpha_.narrow(0, 0, num_sequences)
                                   .narrow(1, t, 1)
                                   .narrow(2, num_hmm_states, 1)
                                   .squeeze(2)
                                   .squeeze(1); // B
    this_alpha_sums.copy_(alpha_t.logsumexp(1)); // B

    if (GetVerboseLevel() >= 3) {
      std::cerr << "alpha @" << t << " = " << alpha_t;
      std::cerr << "alpha-sum @" << t << " = " << this_alpha_sums;
    }
  }
}

torch::Tensor ChainNumeratorComputation::Forward() {
  AlphaFirstFrame();
  AlphaRemainingFrames();
  auto obj = ComputeTotLogLike();
  return obj;
}

torch::Tensor ChainNumeratorComputation::ComputeTotLogLike() {
  auto sequence_lengths_a = sequence_lengths_.accessor<int, 1>();
  if (GetVerboseLevel() >= 3)
    std::cerr << "final probs = " << final_probs_;
  for (int s = 0; s < num_sequences_; s++) {
    int sequence_length = sequence_lengths_a[s];
    torch::Tensor last_frame_alpha = alpha_.narrow(0, s, 1)
      .narrow(1, sequence_length, 1)
      .narrow(2, 0, num_states_).squeeze(1).squeeze(0); // H
    if (GetVerboseLevel() >= 3)
      std::cerr << "last frame alpha for sequence "
                << s << " = " << last_frame_alpha << std::endl;
    torch::Tensor this_final_probs =
        final_probs_.narrow(0, s, 1).squeeze(0); // H
    torch::Tensor last_frame_alpha_sum =
        last_frame_alpha.add(this_final_probs).logsumexp(0); // 1
    if (GetVerboseLevel() >= 3)
      std::cerr << "last frame alpha sum for sequence "
                << s << " = " << last_frame_alpha_sum << std::endl;
    torch::Tensor alpha_frame_sum = alpha_.narrow(0, s, 1)
      .narrow(1, 0, sequence_length)
      .narrow(2, num_states_, 1).squeeze(2).squeeze(0); // T
    if (GetVerboseLevel() >= 3)
      std::cerr << "alpha frame sums for sequence " << s
                << " = " << alpha_frame_sum << std::endl;
    tot_log_prob_.narrow(0, s, 1).copy_(alpha_frame_sum.sum() + last_frame_alpha_sum);
    if (GetVerboseLevel() >= 3)
      std::cerr << "tot_log_prob for sequence " << s
                << " = " << alpha_frame_sum.sum()
                << " + " << last_frame_alpha_sum
                << " = " << tot_log_prob_.narrow(0, s, 1);
  }
  return tot_log_prob_.sum();
}

void ChainNumeratorComputation::BetaLastFrame() {
  auto sequence_lengths_a = sequence_lengths_.accessor<int, 1>();
  for (int s = 0; s < num_sequences_; s++) {
    int sequence_length = sequence_lengths_a[s];
    torch::Tensor last_frame_beta = beta_.narrow(0, s, 1)
      .narrow(1, sequence_length % 2, 1).squeeze(1).squeeze(0); // H
    if (GetVerboseLevel() >= 3)
      std::cerr << "last frame beta for sequence "
                << s << " = " << last_frame_beta << std::endl;
    torch::Tensor last_frame_alpha = alpha_.narrow(0, s, 1)
      .narrow(1, sequence_length, 1)
      .narrow(2, 0, num_states_).squeeze(1).squeeze(0); // H
    if (GetVerboseLevel() >= 3)
      std::cerr << "last frame alpha dash for sequence "
                << s << " = " << last_frame_alpha << std::endl;
    torch::Tensor this_final_probs = final_probs_.narrow(0, s, 1).squeeze(0); // H
    torch::Tensor tot_prob = last_frame_alpha.add(this_final_probs).logsumexp(0); // 1
    last_frame_beta.copy_(this_final_probs);
    last_frame_beta.add_(-tot_prob);
    if (GetVerboseLevel() >= 3)
      std::cerr << "beta @ last-frame for " << s << " = "
                << last_frame_beta << std::endl;
  }
}

void ChainNumeratorComputation::BetaRemainingFrames() {
  auto batch_sizes_a = batch_sizes_.accessor<long, 1>();
  int num_hmm_states = num_states_,
    num_frames = num_frames_,
    num_pdfs = num_pdfs_,
    num_transitions = num_transitions_;

  auto transition_indices_a = forward_transition_indices_.accessor<int, 3>();
  auto transitions_a = forward_transitions_.accessor<int, 3>();
  auto transition_probs_a = forward_transition_probs_.accessor<float, 2>();

  for (int t = num_frames - 1; t >= 0; t--) {
    long num_sequences = batch_sizes_a[t];

    // B x (H + 1)
    torch::Tensor alpha_t = alpha_.narrow(1, t, 1).squeeze(1),
      beta_tp1 = beta_.narrow(1, (t + 1) % 2, 1).squeeze(1),
      beta_t = beta_.narrow(1, t % 2, 1).squeeze(1);
    // 'probs' is the matrix of pseudo-likelihoods for frame t.
    torch::Tensor probs = nnet_output_.narrow(1, t, 1).squeeze(1);  // B x V
    torch::Tensor log_prob_deriv = nnet_output_deriv_.narrow(1, t, 1).squeeze(1);  // B x V

    auto probs_a = probs.accessor<float, 2>();
    auto log_prob_deriv_a = log_prob_deriv.accessor<float, 2>();
    auto alpha_t_a = alpha_t.accessor<float, 2>();
    auto beta_t_a = beta_t.accessor<float, 2>();
    auto beta_tp1_a = beta_tp1.accessor<float, 2>();

    for (int s = 0; s < num_sequences; s++) {
      for (int h = 0; h < num_hmm_states; h++) {
        float this_alpha_prob = alpha_t_a[s][h],
            inv_arbitrary_scale = alpha_t_a[s][num_hmm_states];
        float tot_variable_factor = -std::numeric_limits<float>::infinity();
        for (int trans_i = transition_indices_a[s][h][0];
            trans_i != transition_indices_a[s][h][1]; trans_i++) {
          float transition_prob = transition_probs_a[s][trans_i];
          int pdf_id = transitions_a[s][trans_i][2],
              next_hmm_state = transitions_a[s][trans_i][1];
          float variable_factor = transition_prob +
              beta_tp1_a[s][next_hmm_state] +
              probs_a[s][pdf_id] - inv_arbitrary_scale;
          tot_variable_factor = LogAdd(tot_variable_factor, variable_factor);
          float occupation_prob = variable_factor + this_alpha_prob;
          log_prob_deriv_a[s][pdf_id] = LogAdd(log_prob_deriv_a[s][pdf_id],
                                               occupation_prob);
          //log_prob_deriv_a[s][pdf_id] += Exp(occupation_prob);
        }
        beta_t_a[s][h] = tot_variable_factor;
      }
    }

    if (GetVerboseLevel() >= 3)
      std::cerr << "beta @ " << t << " = " << beta_t << std::endl;
  }
}

bool ChainNumeratorComputation::Backward() {
  BetaLastFrame();
  BetaRemainingFrames();
  return CheckValues();
}

bool ChainNumeratorComputation::CheckValues() {
  auto sequence_lengths_a = sequence_lengths_.accessor<int, 1>();
  for (int s = 0; s < num_sequences_; s++) {
    int sequence_length = sequence_lengths_a[s];
    std::vector<int> times = {sequence_length - 1, 0};

    for (const int t : times) {
      torch::Tensor log_prob_deriv = nnet_output_deriv_
          .narrow(0, s, 1).narrow(1, t, 1).squeeze(1).squeeze(0); // V
      float deriv_sum = log_prob_deriv.exp().sum().cpu().data<float>()[0];

      if (!ApproxEqual(deriv_sum, 1.0)) {
        std::cerr << "On time " << t << " for seq " << s << ", deriv sum "
                  << deriv_sum << " != 1.0" << std::endl;
        if (fabs(deriv_sum - 1.0) > 0.05 || deriv_sum - deriv_sum != 0) {
          std::cerr << "Excessive error detected, will abandon this minibatch"
                    << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}
