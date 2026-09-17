# Build in place so lib/deformcuda/_C*.so sits next to __init__.py:
#   cd lib/deformcuda && python setup.py build_ext --inplace
# Compiled via GitHub Actions (never on the instance), published as a release asset.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="anchorflow_deformcuda",
    ext_modules=[CUDAExtension("_C", ["deformcuda_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
