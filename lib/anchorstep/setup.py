# Build in place so lib/anchorstep/_C*.so sits next to __init__.py:
#   cd lib/anchorstep && python setup.py build_ext --inplace
# Compiled via GitHub Actions (never on the instance), published as a release asset.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="anchorflow_anchorstep",
    ext_modules=[CUDAExtension("_C", ["anchorstep_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
