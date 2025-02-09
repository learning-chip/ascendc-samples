import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from ascendc_extension import AscendCExtension
CURRENT_DIR = os.path.dirname(__file__)

package_name = 'custom_vector_ops'
setup(
    name=package_name,
    ext_modules=[
        AscendCExtension(
            name=package_name,
            sources=['torch_interface.cpp'],
            extra_library_dirs=[
                os.path.join(CURRENT_DIR)
                ],  # location of custom lib{name}.so file
            extra_libraries=[
                'silu_ascendc'
                ]  # name of custom lib{name}.so file
            ),
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
