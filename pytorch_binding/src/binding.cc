#include <torch/torch.h>
#include "chain-den-graph.h"
#include "base.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_verbose_level", &SetVerboseLevel);

  py::class_<chain::DenominatorGraph>(m, "DenominatorGraph")
    .def(py::init<const fst::StdVectorFst&, int32>()) 
    .def("initial_probs", &chain::DenominatorGraph::InitialProbs)
    .def("transitions", &chain::DenominatorGraph::Transitions)
    .def("transition_probs", &chain::DenominatorGraph::TransitionProbs)
    .def("num_states", &chain::DenominatorGraph::NumStates)
    .def("num_pdfs", &chain::DenominatorGraph::NumPdfs);
}
