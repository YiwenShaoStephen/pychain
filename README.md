# PyTorch implementation of LF-MMI for End-to-end ASR

End-to-end version of lattice-free MMI (LF-MMI or chain model) implemented in PyTorch.  
*TODO*:
regular version of LF-MMI.
## What's New:
- GPU computation for both denominator and numerator graphs
- Support unequal length sequences within a minibatch

## Installation and Requirements
* [PyTorch](http://pytorch.org/) version >= 1.4.0

### First-time Installation (including [OpenFST](http://www.openfst.org/twiki/bin/view/FST/FstDownload))
```
git clone https://github.com/YiwenShaoStephen/pychain.git
pip install kaldi_io
make
```

### Update
Whenever you update or modify any none-python codes (e.g. .c or .cu) in pychain, you need to re-compile it by 
```
make pychain
```

## Reference
1. "End-to-end speech recognition using lattice-free MMI", Hossein Hadian, Hossein Sameti, Daniel Povey, Sanjeev Khudanpur, Interspeech 2018 [(pdf)](http://www.danielpovey.com/files/2018_interspeech_end2end.pdf)
2. "Purely sequence-trained neural networks for ASR based on lattice-free MMI", Daniel Povey, Vijayaditya Peddinti, Daniel Galvez, Pegah Ghahrmani, Vimal Manohar, Xingyu Na, Yiming Wang and Sanjeev Khudanpur, Interspeech 2016, [(pdf)](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) [(slides,pptx)](http://www.danielpovey.com/files/2016_interspeech_mmi_presentation.pptx)

The code is based on the original version in [kaldi](https://github.com/kaldi-asr/kaldi) repository with no dependency on the kaldi base.
