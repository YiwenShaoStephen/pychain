# PyTorch implementation of LF-MMI for End-to-end ASR

End-to-end version of lattice-free MMI (LF-MMI or chain model) implemented in PyTorch.
*TODO*:
regular version of LF-MMI.

## Installation

### Setup OpenFst
1. Download and install [OpenFST](http://www.openfst.org/twiki/bin/view/FST/FstDownload)\
* Install it by:
```bash
./configure --prefix=`pwd` --enable-static --enable-shared --enable-ngram-fsts CXX="g++" LIBS="-ldl" CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
make
make install
```
Note that the option `-D_GLIBCXX_USE_CXX11_ABI=0` must be compatible with the 
option used when compiling PyTorch. Details [here](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html).

2. Path
```
export OPENFST_PATH=<your_dir>/openfst;
export LD_LIBRARY_PATH=<your_dir>/openfst/lib:$LD_LIBRARY_PATH;
```

### Install the bindings

```bash
cd openfst_binding
python setup.py install
cd ..
cd pytorch_binding
python setup.py install
cd ..
```

## Reference
1. "End-to-end speech recognition using lattice-free MMI", Hossein Hadian, Hossein Sameti, Daniel Povey, Sanjeev Khudanpur, Interspeech 2018 [(pdf)](http://www.danielpovey.com/files/2018_interspeech_end2end.pdf)
2. "Purely sequence-trained neural networks for ASR based on lattice-free MMI", Daniel Povey, Vijayaditya Peddinti, Daniel Galvez, Pegah Ghahrmani, Vimal Manohar, Xingyu Na, Yiming Wang and Sanjeev Khudanpur, Interspeech 2016, [(pdf)](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) [(slides,pptx)](http://www.danielpovey.com/files/2016_interspeech_mmi_presentation.pptx)

The code is a modification of the version in the [kaldi](https://github.com/kaldi-asr/kaldi) repository with no dependency on the kaldi base.
