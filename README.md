# PyTorch implementation of LF-MMI for End-to-end ASR

End-to-end version of lattice-free MMI is mainly based on these two papers:
1. "End-to-end speech recognition using lattice-free MMI", Hossein Hadian, Hossein Sameti, Daniel Povey, Sanjeev Khudanpur, Interspeech 2018 [(pdf)](http://www.danielpovey.com/files/2018_interspeech_end2end.pdf)
2. "Purely sequence-trained neural networks for ASR based on lattice-free MMI", Daniel Povey, Vijayaditya Peddinti, Daniel Galvez, Pegah Ghahrmani, Vimal Manohar, Xingyu Na, Yiming Wang and Sanjeev Khudanpur, Interspeech 2016, [(pdf)](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) [(slides,pptx)](http://www.danielpovey.com/files/2016_interspeech_mmi_presentation.pptx)

The code is a modification of the version in the [kaldi](https://github.com/kaldi-asr/kaldi) repository with no dependency on the kaldi base.

## Installation

### Install PyTorch
Install [PyTorch](https://github.com/pytorch/pytorch#installation)

### Setup OpenFst
Download and install [OpenFST](http://www.openfst.org/twiki/bin/view/FST/FstDownload)

`OPENFST_PATH` shold be set to the root of the OpenFst installation.
i.e. `$OPENFST_PATH/include` and `$OPENFST_PATH/openfst` should contain the required 
headers and libraries.

```bash
./configure --prefix=`pwd` --enable-static --enable-shared --enable-ngram-fsts CXX="g++" LIBS="-ldl" CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
make
make install
```

Note that the option `-D_GLIBCXX_USE_CXX11_ABI=0` must be compatible with the 
option used when compiling PyTorch. Details [here](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html).

### Install the bindings

```bash
cd pytorch_binding
python setup.py install
```

## Structure of the directory

- `/` contains the instructions and makefiles etc.
  - `src` contains C++/CUDA code agnostic of PyTorch
  - `pytorch_bindings` contains the pytorch-based module of the LF-MMI objf function
    - `src`  contains C++ level implementation of the LF-MMI objf function
    - `pychain_pytorch`  contains python level implementation of the LF-MMI objf function. This includes the actual loss-function, which can be bound the C++ level code.
  - `openfst_bindings` contains the openfst-based module to interact with FSTs
