import os
from setuptools import setup
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

# Fix issue: Build extensions with/without CUDA [#27222](https://github.com/pytorch/pytorch/issues/27222)
# The following code block is borrowed from: https://github.com/microsoft/pytorch_od
assert CUDA_HOME or os.getenv("FORCE_CUDA", "0") == "1", "CUDA not found"
if not torch.cuda.is_available():
      arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
      if not arch_list:
            print("No CUDA runtime found and TORCH_CUDA_ARCH_LIST not set")
            try:
                  driver_version = torch._C._cuda_getDriverVersion()
            except Exception as e:
                  print("torch._C._cuda_getDriverVersion() may be deprecated error: {}".format(e))
                  driver_version = 0
            if driver_version == 0:
                  arch_list = 'Pascal;Volta;Turing'
                  print("No driver found defaulting TORCH_CUDA_ARCH_LIST to {}".format(arch_list))
                  os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list

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
