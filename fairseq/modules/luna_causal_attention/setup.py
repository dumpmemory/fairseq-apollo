# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name='lca',
    ext_modules=[
        CUDAExtension('lca_cuda', [
            'lightconv_cuda.cpp',
            'lightconv_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
