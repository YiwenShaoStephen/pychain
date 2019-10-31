from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

openfst_path = "../openfst"

if "OPENFST_PATH" in os.environ:
    openfst_path = os.environ["OPENFST_PATH"]

if not os.path.exists(os.path.join(openfst_path, "lib", "libfst.so")):
    raise SystemExit("Could not find libfst.so in {}.\n"
                     "Install openfst and set OPENFST_PATH to the openfst "
                     "root directory".format(openfst_path))

setup(name='simplefst',
      ext_modules=[CppExtension('simplefst', ['src/fstext.cc'],
                                include_dirs=['src', os.path.join(
                                    openfst_path, 'include')],
                                library_dirs=[os.path.join(
                                    openfst_path, 'lib')],
                                libraries=['fst', 'fstscript'])],
      cmdclass={'build_ext': BuildExtension})
