// chain-computation.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)
//                 2019  Yiwen Shao

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
#include <torch/extension.h>

/*
  This extended comment describes how we implement forward-backward without log
  and without overflow, and also the leaky-HMM idea.

  We'll start by establishing the notation for conventional forward-backward,
  then add the 'arbitrary-scale' concept that prevents overflow, and then
  add the 'leaky-hmm' concept.

  All this is done in parallel over multiple sequences, but the computations
  are independent over the separate sequences, so we won't introduce any notation
  or index for the sequence; we'll just explain it for one sequence.

  Suppose we have I hmm-states, numbered i = 0 ... I-1 (we'll use i and j for
  hmm-state indexes).  Let foll(i) give a list of arcs leaving state i, and
  pred(i) give a list of arcs entering state i, and we'll use notation like:
    for (j, p, n) in foll(i):
  for iterating over those arcs, where in this case j is the destination-state,
  p is the transition-probability of the arc and n is the pdf-id index.
  We can then look up the emission probability as x(t, n) for some frame
  0 <= t < T.

  ** Version 1 of the computation (naive version) **

  * Forward computation (version 1)

  In the forward computation we're computing alpha(i, t) for 0 <= t <= T):
    - For the first frame, set alpha(0, i) = init(i), where init(i) is the
      initial-probabilitiy from state i.  # in our framework these are obtained
      #  by running the HMM for a while and getting an averaged occupation
      # probability, and using this as an initial-prob, since the boundaries of
      # chunks don't really correspond to utterance boundaries in general.]
    - For t = 1 ... T:
        for i = 0 ... I-1:
           alpha(t, i) = 0
           for (j, p, n) in pred(i):  # note: j is preceding-state.
              alpha(t, i) += x(t-1, n) * alpha(t-1, j) * p.

    - total-prob = \sum_i alpha(T, i).  # note, we take the final-probs of all states
                                        # to be 1.0.

  * Backward computation (version 1)

  And now for the backward computation.  Contrary to tradition, we include the
  inverse of the total-prob as a factor in the betas.  This is both more
  convenient (it simplifies the way we obtain posteriors), and makes the
  algorithm more generalizable as all the beta quantities can be interpreted as
  the partial derivative of the overall logprob with respect to their
  corresponding alpha.

  In forward backward notation, gamma is normally used for state-level
  occupation probabilities, but what we care about here is pdf-id-level
  occupation probabilities (i.e. the partial derivative of the overall logprob
  w.r.t. the logs of the x(t, n) quantities), so we use gamma for that.

    - for the final frame:
       for each i, beta(T, i) = 1 / total-prob.
    - for t = T-1 ... 0:
        for i = 0 ... I-1:
           beta(t, i) = 0
           for (j, p, n) in foll(i):  # note: j is following-state.
              beta(t, i) += x(t, n) * beta(t+1, j) * p.
              gamma(t, n) += alpha(t, i) * x(t, n) * beta(t+1, j) * p.

  ** Version 2 of the computation (renormalized version) **

  Version 1 of the algorithm is susceptible to numeric underflow and overflow,
  due to the limited range of IEEE floating-point exponents.
  Define tot-alpha(t) = \sum_i alpha(t, i).  Then the renormalized version of
  the computation is as above, except whenever the quantity x(t, n) appears,
  we replace it with x(t, n) / tot-alpha(t).  In the algorithm we refer to
  1.0 / tot-alpha(t) as 'arbitrary_scale', because mathematically we can use any
  value here as long as we are consistent and the value only varies with t
  and not with n; we'll always get the same posteriors (gamma).

  When the algorithm outputs log(total-prob) as the total log-probability
  of the HMM, we have to instead return the expression:
    log(total-prob) + \sum_{t=0}^{T-1} \log tot-alpha(t).
  to correct for the scaling of the x values.

  The algorithm is still vulnerable to overflow in the beta computation because
  it's possible that the dominant path could have a very tiny alpha.  However,
  once we introduce the leaky-HMM idea (below), this problem will disappear.

 */

// This does forward-backward in parallel on a number of sequences, using a
// single HMM.
class ChainComputation {
 public:
  //  Constructor
  ChainComputation(torch::Tensor forward_transitions,
		   torch::Tensor forward_transition_indices,
		   torch::Tensor forward_transition_probs,
		   torch::Tensor backward_transitions,
		   torch::Tensor backward_transition_indices,
		   torch::Tensor backward_transition_probs,
		   torch::Tensor initial_probs,
		   torch::Tensor exp_nnet_output,
		   int num_states);

  // Does the forward computation, and returns the total log-like summed over
  // all sequences.  You will have to scale this by any supervision weighting
  // factor, manually.  Note: this log-like will be negated before it
  // is added into the objective function, since this is the denominator
  // computation.
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
  // sum over all alpha for frame t.
  void AlphaSum(int t);

  // done after all the alphas, this function computes and returns the total
  // log-likelihood summed over all the sequences
  torch::Tensor ComputeTotLogLike();

  // sets up the beta for frame t = T.
  void BetaLastFrame();
  // beta computation for 0 <= beta < num_time_steps_.
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
  torch::Tensor initial_probs_;
  
  // The exp() of the nnet output (the exp() avoids us having to
  // exponentiate in the forward-backward).
  torch::Tensor exp_nnet_output_;

  // the derivs w.r.t. the nnet outputs
  torch::Tensor nnet_output_deriv_;

  // dimension is (num-sequences, frames_per_sequence + 1, num-hmm-states + 1)
  // Note, they are not logs. alpha_[:, :, -1]
  // are for the alpha-sums.
  torch::Tensor alpha_;

  // beta in rolling buffer for the backward algorithm. Dimension is
  // (num_sequences, 2, num-hmm-states + 1))
  torch::Tensor beta_;

  // the total probability for each sequence.
  torch::Tensor tot_prob_;

  // the log of tot_prob_.
  torch::Tensor tot_log_prob_;

  // do computation on cuda or not
  bool cuda_;
  
  bool ok_;
};
