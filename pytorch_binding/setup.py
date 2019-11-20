from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='pychain',
      description="PyTorch wrapper for implementation of E2E LFMMI",
      ext_modules=[CUDAExtension('pychain',
                                 ['src/pychain.cc',
                                  'src/base.cc',
                                  'src/chain-kernels.cu',
                                  'src/chain-denominator.cc'],
                                 include_dirs=['src'])],
      cmdclass={'build_ext': BuildExtension})
