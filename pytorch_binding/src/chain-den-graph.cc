// chain/chain-den-graph.cc

// Copyright      2015-2018   Johns Hopkins University (author: Daniel Povey)
//                     2018   Vimal Manohar

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

#include <vector>
#include <iostream>
#include <torch/torch.h>
#include "chain-den-graph.h"

namespace py = pybind11;

namespace chain {

typedef int int32;
typedef long int64;
typedef float BaseFloat;

DenominatorGraph::DenominatorGraph(const fst::StdVectorFst &fst,
                                   int32 num_pdfs):
    num_pdfs_(num_pdfs) {

  if (GetVerboseLevel() > 2)
    py::print("Before initialization, transition-probs=", transition_probs_);
  SetTransitions(fst, num_pdfs);
  if (GetVerboseLevel() > 2)
    py::print("After initialization, transition-probs=", transition_probs_);
  SetInitialProbs(fst);
}

void DenominatorGraph::SetTransitions(const fst::StdVectorFst &fst,
                                      int32 num_pdfs) {
  int32 num_states = fst.NumStates();
  
  std::vector<int64> indices;
  std::vector<BaseFloat> log_probs;

  for (int32 s = 0; s < num_states; s++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      int32 pdf_id = arc.ilabel - 1;
      assert(pdf_id >= 0 && pdf_id < num_pdfs);
      indices.push_back(s);
      indices.push_back(arc.nextstate);
      indices.push_back(pdf_id);
      log_probs.push_back(-arc.weight.Value());
    }
  }

  if (GetVerboseLevel() > 2)
    py::print("indices=", indices);

  assert(indices.size() == log_probs.size() * 3);

  int64 num_transitions = log_probs.size();
  
  transitions_.resize_({3, num_transitions});
  transitions_.copy_(torch::CPU(at::kLong)
    .tensorFromBlob(indices.data(), {num_transitions, 3})
    .transpose(0, 1));
  if (GetVerboseLevel() > 2)
    py::print("transitions=", transitions_);

  transition_probs_.resize_({num_transitions});
  transition_probs_.copy_(torch::CPU(at::kFloat)
      .tensorFromBlob(log_probs.data(), {num_transitions}));
  if (GetVerboseLevel() > 2)
    py::print("transition-probs-before-exp=", transition_probs_);
  transition_probs_.exp_();
  if (GetVerboseLevel() > 2)
    py::print("transition-probs=", transition_probs_);
}

void DenominatorGraph::SetInitialProbs(const fst::StdVectorFst &fst) {
  // we set only the start-state to have probability mass, and then 100
  // iterations of HMM propagation, over which we average the probabilities.
  // initial probs won't end up making a huge difference as we won't be using
  // derivatives from the first few frames, so this isn't 100% critical.
  int32 num_iters = 100;
  int32 num_states = fst.NumStates();

  // we normalize each state so that it sums to one (including
  // final-probs)... this is needed because the 'chain' code doesn't
  // have transition probabilities.
  at::Tensor normalizing_factor = torch::CPU(at::kDouble).zeros(num_states);
  auto nf_a = normalizing_factor.accessor<double, 1>();
 
  for (int32 s = 0; s < num_states; s++) {
    double tot_prob = exp(-fst.Final(s).Value());
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      tot_prob += exp(-aiter.Value().weight.Value());
    }
    assert(tot_prob > 0.0 && tot_prob < 100.0);
    nf_a[s] = 1.0 / tot_prob;
  }

  at::Tensor cur_prob = torch::CPU(at::kDouble).zeros(num_states),
         next_prob = torch::CPU(at::kDouble).zeros(num_states),
         avg_prob = torch::CPU(at::kDouble).zeros(num_states);

  cur_prob[fst.Start()] = 1.0;
  for (int32 iter = 0; iter < num_iters; iter++) {
    avg_prob.add_(cur_prob, 1.0 / num_iters);
    auto cur_prob_a = cur_prob.accessor<double, 1>();
    auto next_prob_a = next_prob.accessor<double, 1>();
    auto nf_a = normalizing_factor.accessor<double, 1>();

    for (int32 s = 0; s < num_states; s++) {
      double prob = cur_prob_a[s] * nf_a[s];
      for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s); !aiter.Done();
           aiter.Next()) {
        const fst::StdArc &arc = aiter.Value();
        next_prob_a[arc.nextstate] += prob * exp(-arc.weight.Value());
      }
    }
    cur_prob.copy_(next_prob);
    next_prob.zero_();
    // Renormalize, beause the HMM won't sum to one even after the
    // previous normalization (due to final-probs).
    cur_prob.mul_(at::Scalar(1.0 / cur_prob.sum()));
  }

  initial_probs_.resize_(num_states);
  initial_probs_.copy_(avg_prob.toType(torch::CPU(at::kFloat)));
}

void DenominatorGraph::GetNormalizationFst(const fst::StdVectorFst &ifst,
                                           fst::StdVectorFst *ofst) {
  assert(ifst.NumStates() == initial_probs_.size(0));
  if (&ifst != ofst)
    *ofst = ifst;
  int32 new_initial_state = ofst->AddState();
  at::Tensor initial_probs = initial_probs_.to(at::kCPU);
  auto initial_probs_a = initial_probs.accessor<float, 1>();

  for (int32 s = 0; s < initial_probs_.size(0); s++) {
    BaseFloat initial_prob = initial_probs_a[s];
    assert(initial_prob > 0.0);
    fst::StdArc arc(0, 0, fst::TropicalWeight(-log(initial_prob)), s);
    ofst->AddArc(new_initial_state, arc);
    ofst->SetFinal(s, fst::TropicalWeight::One());
  }
  ofst->SetStart(new_initial_state);
  fst::RmEpsilon(ofst);
  fst::ArcSort(ofst, fst::ILabelCompare<fst::StdArc>());
}

/*
void MapFstToPdfIdsPlusOne(const TransitionModel &trans_model,
                           fst::StdVectorFst *fst) {
  int32 num_states = fst->NumStates();
  for (int32 s = 0; s < num_states; s++) {
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(fst, s);
         !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      assert(arc.ilabel == arc.olabel);
      if (arc.ilabel > 0) {
        arc.ilabel = trans_model.TransitionIdToPdf(arc.ilabel) + 1;
        arc.olabel = arc.ilabel;
        aiter.SetValue(arc);
      }
    }
  }
}
*/

void MinimizeAcceptorNoPush(fst::StdVectorFst *fst) {
  BaseFloat delta = fst::kDelta * 10.0;  // use fairly loose delta for
                                         // aggressive minimimization.
  fst::ArcMap(fst, fst::QuantizeMapper<fst::StdArc>(delta));
  fst::EncodeMapper<fst::StdArc> encoder(fst::kEncodeLabels | fst::kEncodeWeights,
                                         fst::ENCODE);
  fst::Encode(fst, &encoder);
  fst::internal::AcceptorMinimize(fst);
  fst::Decode(fst, encoder);
}

// This static function, used in CreateDenominatorFst, sorts an
// fst's states in decreasing order of number of transitions (into + out of)
// the state.  The aim is to have states that have a lot of transitions
// either into them or out of them, be numbered earlier, so hopefully
// they will be scheduled first and won't delay the computation
static void SortOnTransitionCount(fst::StdVectorFst *fst) {
  // negative_num_transitions[i] will contain (before sorting), the pair
  // ( -(num-transitions-into(i) + num-transition-out-of(i)), i)
  int32 num_states = fst->NumStates();
  std::vector<std::pair<int32, int32> > negative_num_transitions(num_states);
  for (int32 i = 0; i < num_states; i++) {
    negative_num_transitions[i].first = 0;
    negative_num_transitions[i].second = i;
  }
  for (int32 i = 0; i < num_states; i++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(*fst, i); !aiter.Done();
         aiter.Next()) {
      negative_num_transitions[i].first--;
      negative_num_transitions[aiter.Value().nextstate].first--;
    }
  }
  std::sort(negative_num_transitions.begin(), negative_num_transitions.end());
  std::vector<fst::StdArc::StateId> order(num_states);
  for (int32 i = 0; i < num_states; i++)
    order[negative_num_transitions[i].second] = i;
  fst::StateSort(fst, order);
}

/*
void DenGraphMinimizeWrapper(fst::StdVectorFst *fst) {
  for (int32 i = 1; i <= 3; i++) {
    fst::StdVectorFst fst_reversed;
    fst::Reverse(*fst, &fst_reversed);
    fst::PushSpecial(&fst_reversed, fst::kDelta * 0.01);
    MinimizeAcceptorNoPush(&fst_reversed);
    fst::Reverse(fst_reversed, fst);
    std::cout << "Number of states and arcs in transition-id FST after reversed "
              << "minimization is " << fst->NumStates() << " and "
              << NumArcs(*fst) << " (pass " << i << ")";
    fst::PushSpecial(fst, fst::kDelta * 0.01);
    MinimizeAcceptorNoPush(fst);
    std::cout << "Number of states and arcs in transition-id FST after regular "
              << "minimization is " << fst->NumStates() << " and "
              << NumArcs(*fst) << " (pass " << i << ")";
  }
  fst::RmEpsilon(fst);
  std::cout << "Number of states and arcs in transition-id FST after "
            << "removing any epsilons introduced by reversal is "
            << fst->NumStates() << " and "
            << NumArcs(*fst);
  fst::PushSpecial(fst, fst::kDelta * 0.01);
}


static void PrintDenGraphStats(const fst::StdVectorFst &den_graph) {
  int32 num_states = den_graph.NumStates();
  int32 degree_cutoff = 3;  // track states with <= transitions in/out.
  int32 num_states_low_degree_in = 0,
      num_states_low_degree_out = 0,
      tot_arcs = 0;
  std::vector<int32> num_in_arcs(num_states, 0);
  for (int32 s = 0; s < num_states; s++) {
    if (den_graph.NumArcs(s) <= degree_cutoff) {
      num_states_low_degree_out++;
    }
    tot_arcs += den_graph.NumArcs(s);
    for (fst::ArcIterator<fst::StdVectorFst> aiter(den_graph, s);
         !aiter.Done(); aiter.Next()) {
      int32 dest_state = aiter.Value().nextstate;
      num_in_arcs[dest_state]++;
    }
  }
  for (int32 s = 0; s < num_states; s++) {
    if (num_in_arcs[s] <= degree_cutoff) {
      num_states_low_degree_in++;
    }
  }
  std::cout << "Number of states is " << num_states << " and arcs "
            << tot_arcs << "; number of states with in-degree <= "
            << degree_cutoff << " is " << num_states_low_degree_in
            << " and with out-degree <= " << degree_cutoff
            << " is " << num_states_low_degree_out;
}
*/

// Check that every pdf is seen, warn if some are not.
static void CheckDenominatorFst(int32 num_pdfs,
                                const fst::StdVectorFst &den_fst) {
  std::vector<bool> pdf_seen(num_pdfs);
  int32 num_states = den_fst.NumStates();
  for (int32 s = 0; s < num_states; s++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(den_fst, s);
         !aiter.Done(); aiter.Next()) {
      int32 pdf_id = aiter.Value().ilabel - 1;
      assert(pdf_id >= 0 && pdf_id < num_pdfs);
      pdf_seen[pdf_id] = true;
    }
  }
  for (int32 pdf = 0; pdf < num_pdfs; pdf++) {
    if (!pdf_seen[pdf]) {
      std::cerr << "Pdf-id " << pdf << " is not seen in denominator graph.";
    }
  }
}

/*
void CreateDenominatorFst(const ContextDependency &ctx_dep,
                          const TransitionModel &trans_model,
                          const fst::StdVectorFst &phone_lm_in,
                          fst::StdVectorFst *den_fst) {
  using fst::StdVectorFst;
  using fst::StdArc;
  assert(phone_lm_in.NumStates() != 0);
  fst::StdVectorFst phone_lm(phone_lm_in);

  std::cout << "Number of states and arcs in phone-LM FST is "
            << phone_lm.NumStates() << " and " << NumArcs(phone_lm);

  int32 subsequential_symbol = trans_model.GetPhones().back() + 1;
  if (ctx_dep.CentralPosition() != ctx_dep.ContextWidth() - 1) {
    // note: this function only adds the subseq symbol to the input of what was
    // previously an acceptor, so we project, i.e. copy the ilabels to the
    // olabels
    AddSubsequentialLoop(subsequential_symbol, &phone_lm);
    fst::Project(&phone_lm, fst::PROJECT_INPUT);
  }
  std::vector<int32> disambig_syms;  // empty list of diambiguation symbols.

  // inv_cfst will be expanded on the fly, as needed.
  fst::InverseContextFst inv_cfst(subsequential_symbol,
                                  trans_model.GetPhones(),
                                  disambig_syms,
                                  ctx_dep.ContextWidth(),
                                  ctx_dep.CentralPosition());

  fst::StdVectorFst context_dep_lm;
  fst::ComposeDeterministicOnDemandInverse(phone_lm, &inv_cfst,
                                           &context_dep_lm);

  // at this point, context_dep_lm will have indexes into 'ilabels' as its
  // input symbol (representing context-dependent phones), and phones on its
  // output.  We don't need the phones, so we'll project.
  fst::Project(&context_dep_lm, fst::PROJECT_INPUT);

  std::cout << "Number of states and arcs in context-dependent LM FST is "
            << context_dep_lm.NumStates() << " and " << NumArcs(context_dep_lm);

  std::vector<int32> disambig_syms_h; // disambiguation symbols on input side
  // of H -- will be empty.
  HTransducerConfig h_config;
  // the default is 1, but just document that we want this to stay as one.
  // we'll use the same value in test time.  Consistency is the key here.
  h_config.transition_scale = 1.0;

  StdVectorFst *h_fst = GetHTransducer(inv_cfst.IlabelInfo(),
                                       ctx_dep,
                                       trans_model,
                                       h_config,
                                       &disambig_syms_h);
  assert(disambig_syms_h.empty());
  StdVectorFst transition_id_fst;
  TableCompose(*h_fst, context_dep_lm, &transition_id_fst);
  delete h_fst;

  BaseFloat self_loop_scale = 1.0;  // We have to be careful to use the same
                                    // value in test time.
  // 'reorder' must always be set to true for chain models.
  bool reorder = true;
  bool check_no_self_loops = true;

  // add self-loops to the FST with transition-ids as its labels.
  AddSelfLoops(trans_model, disambig_syms_h, self_loop_scale, reorder,
               check_no_self_loops, &transition_id_fst);
  // at this point transition_id_fst will have transition-ids as its ilabels and
  // context-dependent phones (indexes into IlabelInfo()) as its olabels.
  // Discard the context-dependent phones by projecting on the input, keeping
  // only the transition-ids.
  fst::Project(&transition_id_fst, fst::PROJECT_INPUT);

  MapFstToPdfIdsPlusOne(trans_model, &transition_id_fst);
  std::cout << "Number of states and arcs in transition-id FST is "
            << transition_id_fst.NumStates() << " and "
            << NumArcs(transition_id_fst);

  // RemoveEpsLocal doesn't remove all epsilons, but it keeps the graph small.
  fst::RemoveEpsLocal(&transition_id_fst);
  // If there are remaining epsilons, remove them.
  fst::RmEpsilon(&transition_id_fst);
  std::cout << "Number of states and arcs in transition-id FST after "
            << "removing epsilons is "
            << transition_id_fst.NumStates() << " and "
            << NumArcs(transition_id_fst);

  DenGraphMinimizeWrapper(&transition_id_fst);

  SortOnTransitionCount(&transition_id_fst);

  *den_fst = transition_id_fst;
  CheckDenominatorFst(trans_model.NumPdfs(), *den_fst);
  PrintDenGraphStats(*den_fst);
}
*/

int32 DenominatorGraph::NumStates() const {
  return initial_probs_.size(0);
}

}
