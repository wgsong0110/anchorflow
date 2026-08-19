# Build in place so lib/mpmstep/_C*.so sits next to __init__.py:
#   cd lib/mpmstep && python setup.py build_ext --inplace
# Compiled via GitHub Actions (never on the instance), published as a release asset.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="anchorflow_mpmstep",
    ext_modules=[CUDAExtension("_C", ["mpmstep_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
