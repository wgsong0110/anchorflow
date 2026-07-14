from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="anchorflow_lbs",
    ext_modules=[CUDAExtension("anchorflow_lbs._C", ["lbs_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
