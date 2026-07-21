from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="anchorflow_ssm",
    ext_modules=[CUDAExtension("_C", ["ssm_step.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
