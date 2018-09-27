#include <pybind11/pybind11.h>
#include <fst/fstlib.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<fst::StdVectorFst>(m, "StdVectorFst")
    .def(py::init())
    .def("write", (bool (fst::StdVectorFst::*)(const std::string &) const) &fst::StdVectorFst::Write)
    .def_static("read", (fst::StdVectorFst* (*)(const std::string &)) &fst::StdVectorFst::Read)
    .def("num_states", &fst::StdVectorFst::NumStates);
}
