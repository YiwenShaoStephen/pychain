from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='pychain_C',
      description="PyTorch wrapper for implementation of LFMMI",
      ext_modules=[CUDAExtension('pychain_C',
                                 ['src/pychain.cc',
                                  'src/base.cc',
                                  'src/chain-kernels.cu',
                                  'src/chain-log-domain-kernels.cu',
                                  'src/chain-computation.cc',
                                  'src/chain-log-domain-computation.cc'],
                                 include_dirs=['src'])],
      cmdclass={'build_ext': BuildExtension})
