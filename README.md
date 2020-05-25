# PyTorch implementation of LF-MMI for End-to-end ASR

End-to-end version of lattice-free MMI (LF-MMI or chain model) implemented in PyTorch.  
*TODO*:
regular version of LF-MMI.
## What's New:
- GPU computation for both denominator and numerator graphs
- Support unequal length sequences within a minibatch
- Examples of using PyChain: [Espresso](https://github.com/freewym/espresso) and [pychain-example](https://github.com/YiwenShaoStephen/pychain_example)

## Installation and Requirements
* [PyTorch](http://pytorch.org/) version >= 1.4.0

### First-time Installation (including [OpenFST](http://www.openfst.org/twiki/bin/view/FST/FstDownload))
```
pip install kaldi_io
git clone https://github.com/YiwenShaoStephen/pychain.git
cd pychain
make
```

### Update
Whenever you update or modify any none-python codes (e.g. .c or .cu) in pychain, you need to re-compile it by 
```
make pychain
```

## Reference
"PyChain: A Fully Parallelized PyTorch Implementation of LF-MMI for End-to-End ASR", Yiwen Shao, Yiming Wang, Daniel Povey and Sanjeev Khudanpur [(pdf)](https://arxiv.org/abs/2005.09824)
