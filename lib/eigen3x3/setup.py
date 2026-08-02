# Build in place so lib/eigen3x3/_C*.so sits next to __init__.py (import eigen3x3 -> eigen3x3._C):
#   cd lib/eigen3x3 && python setup.py build_ext --inplace
# Compiled via GitHub Actions (never on the instance), published as a release asset.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="anchorflow_eigen3x3",
    ext_modules=[CUDAExtension("_C", ["eigen3x3_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
