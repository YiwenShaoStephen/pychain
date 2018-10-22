from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

openfst_path = "../openfst"

if "OPENFST_PATH" in os.environ:
    openfst_path = os.environ["OPENFST_PATH"]

if not os.path.exists(os.path.join(openfst_path, "lib", "libfst.so")):
    raise SystemExit("Could not find libfst.so in {}.\n"
                     "Install openfst and set OPENFST_PATH to the openfst "
                     "root directory".format(openfst_path))

# setup(name='denominator_graph',
#      ext_modules=[CppExtension('denominator_graph', ['src/base.cc', 'src/chain-den-graph.cc', 'src/binding.cc'],
#                                include_dirs=['src', os.path.join(openfst_path, 'include')],
#                                library_dirs=[os.path.join(openfst_path, 'lib')],
#                                libraries=['fst', 'fstscript'])],
#      cmdclass={'build_ext': BuildExtension})
#
setup(name='pychain',
      description="PyTorch wrapper for implementation of E2E LFMMI",
      ext_modules=[CUDAExtension('pychain',
                                 ['src/base.cc',
                                  'src/chain-den-graph.cc',
                                  'src/chain-denominator.cc',
                                  'src/chain-training.cc',
                                  'src/binding.cc',
                                  'src/chain-kernels.cu'],
                                 include_dirs=['src', os.path.join(
                                     openfst_path, 'include')],
                                 library_dirs=[os.path.join(
                                     openfst_path, 'lib')],
                                 libraries=['fst', 'fstscript'])],
      cmdclass={'build_ext': BuildExtension})
