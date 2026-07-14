# Build in place so lib/lbs/_C*.so sits next to __init__.py (import lbs -> lbs._C):
#   cd lib/lbs && python setup.py build_ext --inplace
# Compiled in the anchorflow image via .github/workflows/cuda-build.yml (never on
# the instance) and published as a release asset; instance_setup.sh downloads it.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="anchorflow_lbs",
    ext_modules=[CUDAExtension("_C", ["lbs_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
