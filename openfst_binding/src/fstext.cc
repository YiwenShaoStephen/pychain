#include <torch/extension.h>
#include <fst/fstlib.h>

namespace py = pybind11;

namespace fst {
  VectorFst<StdArc> *ReadFstFromArk(const std::string & filename, int offset) {
    FstHeader hdr;
    std::ifstream is (filename, std::ifstream::binary);
    // seek to the pos(offset)
    is.seekg(offset, std::ios_base::beg);
    hdr.Read(is, filename);
    FstReadOptions ropts("<unspecified>", &hdr);
    VectorFst<StdArc> *fst = VectorFst<StdArc>::Read(is, ropts);
    return fst;
  }
}

std::vector<torch::Tensor> FstToTensor(const fst::StdVectorFst &fst, bool log_domain=false) {
  struct GraphTransition {
    int in_state;
    int out_state;
    int pdf_id;
    float log_prob;

    GraphTransition(int is, int os, int pdf, float log_prob):
      in_state(is), out_state(os), pdf_id(pdf), log_prob(log_prob) { }
  };

  int num_states = fst.NumStates();

  std::vector<std::vector<GraphTransition> > transitions_out_tup(num_states);
  std::vector<std::vector<GraphTransition> > transitions_in_tup(num_states);
  std::vector<float> final_probs(num_states);

  for (int s = 0; s < num_states; s++) {
    final_probs[s] = -fst.Final(s).Value();
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      int pdf_id = arc.ilabel - 1;
      assert(pdf_id >= 0 && pdf_id < num_pdfs);
      transitions_out_tup[s].emplace_back(s, arc.nextstate, pdf_id, -arc.weight.Value());
      transitions_in_tup[arc.nextstate].emplace_back(s, arc.nextstate, pdf_id, -arc.weight.Value());
    }
  }

  std::vector<int> forward_transition_indices(2 * num_states);
  std::vector<int> forward_transitions;
  std::vector<float> forward_log_probs;
  for (int s = 0; s < num_states; s++) {
    forward_transition_indices[2*s] = static_cast<int32>(forward_transitions.size()) / 3;
    for (auto it = transitions_out_tup[s].begin();
         it != transitions_out_tup[s].end(); ++it) {
      auto& transition = *it;
      forward_transitions.push_back(transition.in_state);
      forward_transitions.push_back(transition.out_state);
      forward_transitions.push_back(transition.pdf_id);
      forward_log_probs.push_back(transition.log_prob);
    }
    forward_transition_indices[2*s+1] = static_cast<int32>(forward_transitions.size()) / 3;
  }

  std::vector<int> backward_transition_indices(2 * num_states);
  std::vector<int> backward_transitions;
  std::vector<float> backward_log_probs;
  for (int s = 0; s < num_states; s++) {
    backward_transition_indices[2*s] = static_cast<int32>(backward_transitions.size()) / 3;
    for (auto it = transitions_in_tup[s].begin();
         it != transitions_in_tup[s].end(); ++it) {
      auto& transition = *it;
      backward_transitions.push_back(transition.in_state);
      backward_transitions.push_back(transition.out_state);
      backward_transitions.push_back(transition.pdf_id);
      backward_log_probs.push_back(transition.log_prob);
    }
    backward_transition_indices[2*s+1] = static_cast<int32>(backward_transitions.size()) / 3;
  }
  int num_transitions = forward_log_probs.size();

  torch::Tensor forward_transitions_tensor = torch::empty({num_transitions, 3}, torch::kInt);
  forward_transitions_tensor.copy_(torch::from_blob(forward_transitions.data(), {num_transitions, 3}, torch::kInt));

  torch::Tensor forward_transition_indices_tensor = torch::empty({num_states, 2}, torch::kInt);
  forward_transition_indices_tensor.copy_(torch::from_blob(forward_transition_indices.data(), {num_states, 2}, torch::kInt));

  torch::Tensor forward_transition_probs_tensor = torch::empty({num_transitions}, torch::kFloat);
  forward_transition_probs_tensor.copy_(torch::from_blob(forward_log_probs.data(), {num_transitions}, torch::kFloat));
  if (!log_domain)
    forward_transition_probs_tensor.exp_();

  torch::Tensor backward_transitions_tensor = torch::empty({num_transitions, 3}, torch::kInt);
  backward_transitions_tensor.copy_(
    torch::from_blob(backward_transitions.data(), {num_transitions, 3}, torch::kInt));

  torch::Tensor backward_transition_indices_tensor = torch::empty({num_states, 2}, torch::kInt);
  backward_transition_indices_tensor.copy_(torch::from_blob(backward_transition_indices.data(), {num_states, 2}, torch::kInt));

  torch::Tensor backward_transition_probs_tensor = torch::empty({num_transitions}, torch::kFloat);
  backward_transition_probs_tensor.copy_(torch::from_blob(backward_log_probs.data(), {num_transitions}, torch::kFloat));
  if (!log_domain)
    backward_transition_probs_tensor.exp_();

  torch::Tensor final_probs_tensor = torch::empty({num_states}, torch::kFloat);
  final_probs_tensor.copy_(torch::from_blob(final_probs.data(), {num_states}, torch::kFloat));
  if (!log_domain)
    final_probs_tensor.exp_();

  return {forward_transitions_tensor,
      forward_transition_probs_tensor,
      forward_transition_indices_tensor,
      backward_transitions_tensor,
      backward_transition_probs_tensor,
      backward_transition_indices_tensor,
      final_probs_tensor
      };
}


torch::Tensor SetLeakyProbs(const fst::StdVectorFst &fst) {
  // we set only the start-state to have probability mass, and then 100
  // iterations of HMM propagation, over which we average the probabilities.
  // initial probs won't end up making a huge difference as we won't be using
  // derivatives from the first few frames, so this isn't 100% critical.
  int32 num_iters = 100;
  int32 num_states = fst.NumStates();

  // we normalize each state so that it sums to one (including
  // final-probs)... this is needed because the 'chain' code doesn't
  // have transition probabilities.
  torch::Tensor normalizing_factor = torch::zeros(num_states, torch::kDouble);
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

  torch::Tensor cur_prob = torch::zeros(num_states, torch::kDouble),
    next_prob = torch::zeros(num_states, torch::kDouble),
    avg_prob = torch::zeros(num_states, torch::kDouble);

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
    cur_prob.mul_(1.0 / cur_prob.sum());
  }

  return avg_prob.toType(torch::kFloat);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<fst::StdVectorFst>(m, "StdVectorFst")
    .def(py::init())
    .def("write", (bool (fst::StdVectorFst::*)(const std::string &) const) &fst::StdVectorFst::Write)
    .def_static("read", (fst::StdVectorFst* (*)(const std::string &)) &fst::StdVectorFst::Read)
    .def_static("read_ark", &fst::ReadFstFromArk)
    .def_static("fst_to_tensor", &FstToTensor)
    .def_static("set_leaky_probs", &SetLeakyProbs)
    .def("num_states", &fst::StdVectorFst::NumStates)
    .def("start_state", &fst::StdVectorFst::Start);
}
