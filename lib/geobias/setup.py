# Build in place so lib/geobias/_C*.so sits next to __init__.py:
#   cd lib/geobias && python setup.py build_ext --inplace
# Compiled via GitHub Actions (never on the instance), published as a release asset.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="anchorflow_geobias",
    ext_modules=[CUDAExtension("_C", ["geobias_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
