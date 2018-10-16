#include <torch/torch.h>
#include "chain-den-graph.h"
#include "chain-training.h"
#include "base.h"
#include "pybind11/iostream.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::add_ostream_redirect(m, "ostream_redirect");

  m.def("set_verbose_level", &SetVerboseLevel);

  py::class_<chain::DenominatorGraph>(m, "DenominatorGraph")
    .def(py::init<const fst::StdVectorFst&, int32>()) 
    .def("initial_probs", &chain::DenominatorGraph::InitialProbs)
    .def("forward_transitions", &chain::DenominatorGraph::ForwardTransitions)
    .def("forward_transition_probs", &chain::DenominatorGraph::ForwardTransitionProbs)
    .def("forward_transition_indices", &chain::DenominatorGraph::ForwardTransitionIndices)
    .def("backward_transitions", &chain::DenominatorGraph::BackwardTransitions)
    .def("backward_transition_probs", &chain::DenominatorGraph::BackwardTransitionProbs)
    .def("backward_transition_indices", &chain::DenominatorGraph::BackwardTransitionIndices)
    .def("num_states", &chain::DenominatorGraph::NumStates)
    .def("num_pdfs", &chain::DenominatorGraph::NumPdfs);
 
  py::class_<chain::ChainTrainingOptions>(m, "ChainTrainingOptions")
    .def(py::init())
    .def_readwrite("l2_regularize", &chain::ChainTrainingOptions::l2_regularize)
    .def_readwrite("xent_regularize", &chain::ChainTrainingOptions::xent_regularize)
    .def_readwrite("leaky_hmm_coefficient", &chain::ChainTrainingOptions::leaky_hmm_coefficient);
  
  m.def("compute_objf_and_deriv", &chain::ComputeObjfAndDeriv);
}
