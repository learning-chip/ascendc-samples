import os
import platform
import torch_npu
from setuptools import Extension
from torch_npu.utils.cpp_extension import TorchExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
PLATFORM_ARCH = platform.machine() + "-linux"

def AscendCExtension(name, sources, extra_library_dirs, extra_libraries):
    kwargs = {}
    cann_home = os.environ['ASCEND_TOOLKIT_HOME']
    include_dirs = [
        os.path.join(cann_home, PLATFORM_ARCH, 'include'),
        os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include')
    ]
    include_dirs.extend(TorchExtension.include_paths())
    kwargs['include_dirs'] = include_dirs

    library_dirs = [
        os.path.join(cann_home, PLATFORM_ARCH, 'lib64'),
        os.path.join(PYTORCH_NPU_INSTALL_PATH, 'lib'),
    ]
    library_dirs.extend(extra_library_dirs)
    library_dirs.extend(TorchExtension.library_paths())
    kwargs['library_dirs'] = library_dirs

    libraries = [
        'c10', 'torch', 'torch_cpu', 'torch_npu', 'torch_python', 'ascendcl',
    ]
    libraries.extend(extra_libraries)
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    return Extension(name, sources, **kwargs)
