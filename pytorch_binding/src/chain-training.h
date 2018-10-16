// chain/chain-training.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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

#include "base.h"
#include "chain-denominator.h"

#ifndef CHAIN_CHAIN_TRAINING_H_
#define CHAIN_CHAIN_TRAINING_H_

namespace chain {

struct ChainTrainingOptions {
  // l2 regularization constant on the 'chain' output; the actual term added to
  // the objf will be -0.5 times this constant times the squared l2 norm.
  // (squared so it's additive across the dimensions).  e.g. try 0.0005.
  BaseFloat l2_regularize;

  // Coefficient for 'leaky hmm'.  This means we have an epsilon-transition from
  // each state to a special state with probability one, and then another
  // epsilon-transition from that special state to each state, with probability
  // leaky_hmm_coefficient times [initial-prob of destination state].  Imagine
  // we make two copies of each state prior to doing this, version A and version
  // B, with transition from A to B, so we don't have to consider epsilon loops-
  // or just imagine the coefficient is small enough that we can ignore the
  // epsilon loops.
  // Note: we generally set leaky_hmm_coefficient to 0.1.
  BaseFloat leaky_hmm_coefficient;


  // Cross-entropy regularization constant.  (e.g. try 0.1).  If nonzero,
  // the network is expected to have an output named 'output-xent', which
  // should have a softmax as its final nonlinearity.
  BaseFloat xent_regularize;

  ChainTrainingOptions(): l2_regularize(0.0), leaky_hmm_coefficient(1.0e-05),
                          xent_regularize(0.0) { }
};

BaseFloat ComputeObjfAndDeriv(const ChainTrainingOptions &opts,
                    const DenominatorGraph &den_graph,
                    int32 num_sequences,
                    at::Tensor nnet_output,
                    at::Tensor nnet_output_deriv);

}  // namespace chain

#endif  // CHAIN_CHAIN_TRAINING_H_
