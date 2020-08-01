// chain-log-domain-computation.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)
//                 2019  Yiwen Shao
//                 2020  Yiming Wang
//                 2020  Facebook Inc.  (Author: Vimal Manohar)

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


#include <vector>
#include <map>
#include <torch/extension.h>  // @manual=//caffe2:torch_extension

/*
   This is a modified forward-backward algorithm from the one implemented in
   chain-computation.h. The main difference is that this one is doing the
   computation in the log-probability domain rather than the original probability
   domain. Leaky HMM mechanism is not implemented as this code is usually to be
   used for computing the numerator part in LF-MMI. see "version 2" in the extended
   comment in chain-computation.h for more detailed descriptions of the algorithm.
 */

// This does forward-backward in parallel on a number of sequences, using a
// single HMM.
class ChainLogDomainComputation {
 public:
  //  Constructor
  ChainLogDomainComputation(
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
    int num_states);

  // Does the forward computation, and returns the total log-like summed over
  // all sequences.  You will have to scale this by any supervision weighting
  // factor, manually.
  torch::Tensor Forward();

  torch::Tensor GetAlpha() {return alpha_;}
  torch::Tensor GetNnetGrad() {return nnet_output_deriv_;}

  // this adds the derivative of the log-prob w.r.t. the
  // nnet output to 'nnet_output_deriv'.
  // returns true if everything seemed OK, false if a failure was detected.
  bool Backward();

 private:
  // sets up the alpha for frame t = 0.
  void AlphaFirstFrame();
  // the alpha computation for some 0 < t <= num_time_steps_.
  void AlphaGeneralFrame(int t);

  // done after all the alphas, this function computes and returns the total
  // log-likelihood summed over all the sequences
  torch::Tensor ComputeTotLogLike();

  // sets up the beta for frame t = T.
  void BetaLastFrame();
  // beta computation for 0 <= t < num_time_steps_.
  void BetaGeneralFrame(int t);

  // some checking that we can do if debug mode is activated, or on frame zero.
  // Sets ok_ to false if a bad problem is detected.
  void BetaGeneralFrameDebug(int t);

  // number of separate frame sequences
  int num_sequences_;
  // number of frames per sequence.
  int num_frames_;
  // number of hmm states
  int num_states_;
  // number of pdf ids
  int num_pdfs_;
  // number of transitions
  int num_transitions_;

  torch::Tensor forward_transitions_;
  torch::Tensor forward_transition_indices_;
  torch::Tensor forward_transition_probs_;
  torch::Tensor backward_transitions_;
  torch::Tensor backward_transition_indices_;
  torch::Tensor backward_transition_probs_;
  // Dimension is (num_sequences, num-hmm-states).
  torch::Tensor initial_probs_;
  torch::Tensor final_probs_;
  torch::Tensor start_state_;

  // The nnet output
  torch::Tensor nnet_output_;

  // batch size of each time step
  torch::Tensor batch_sizes_;

  // sequence_length (i.e. num of frames) of each sequence
  torch::Tensor sequence_lengths_;

  // the derivs w.r.t. the nnet outputs
  torch::Tensor nnet_output_deriv_;

  // the (temporarily) alpha and (more permanently) alpha-dash probabilities;
  // dimension is (num-sequences, frames_per_sequence + 1, num-hmm-states + 1)
  // Note, they are not logs. alpha_[:, :, -1]
  // are for the alpha-sums.
  torch::Tensor alpha_;

  // the beta (also beta-dash) probabilities (rolling buffer) for the backward
  // algorithm. Dimension is (num_sequences, 2, num-hmm-states).
  // Note: for efficiency and to simplify the equations, these are actually the
  // beta / tot_prob_.
  torch::Tensor beta_;

  // the log of tot_prob_.
  torch::Tensor tot_log_prob_;

  // do computation on cuda or not
  bool cuda_;

  bool ok_;
};
