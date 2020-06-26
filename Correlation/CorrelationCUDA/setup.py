from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
      name="Corr2DCUDA",
      py_modules=["Corr2D"], 
      ext_modules=[
            CUDAExtension("Corr2D_ext", [
                  'Corr2D.cpp',
                  'Corr2D_Kernel.cu',
                  ] )
            ],
      cmdclass={
            'build_ext': BuildExtension
            }
      )
