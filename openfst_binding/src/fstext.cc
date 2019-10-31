#include <pybind11/pybind11.h>
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<fst::StdVectorFst>(m, "StdVectorFst")
    .def(py::init())
    .def("write", (bool (fst::StdVectorFst::*)(const std::string &) const) &fst::StdVectorFst::Write)
    .def_static("read", (fst::StdVectorFst* (*)(const std::string &)) &fst::StdVectorFst::Read)
    .def_static("read_ark", &fst::ReadFstFromArk) 
    .def("num_states", &fst::StdVectorFst::NumStates);
}
