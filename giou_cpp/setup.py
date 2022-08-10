import os
import sys
import subprocess

from setuptools import find_packages, setup
from setuptools.command.install import install
# TODO: This is a bit buggy since it requires torch before installing torch.
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from distutils.core import Extension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


class PostInstallation(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Note: buggy for kornia==0.5.3 and it will be fixed in the next version.
        # Set kornia to 0.5.2 temporarily
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'kornia==0.5.2', '--no-dependencies'])


if __name__ == '__main__':
    version = '0.0.1'
    write_version_to_file(version, 'zh_bev_iou/version.py')

    setup(
        name='zh_beviou',
        version=version,
        description='zh_beviou is a test codebase for 3D object bev-iou calculation from point cloud',
        install_requires=[
            'numpy',
            'torch>=1.1',
        ],
        author='Sean Wade',
        author_email='954217436@qq.com',
        license='Apache License 2.0',
        packages=find_packages(),    # exclude=['']
        cmdclass={
            'build_ext': BuildExtension,
            'install': PostInstallation,
            # Post installation cannot be done. ref: https://github.com/pypa/setuptools/issues/1936.
            # 'develop': PostInstallation,
        },
        ext_modules=[
            make_cuda_ext(
                name='beviou_cpu',
                module='zh_bev_iou',
                sources=[
                    'src/beviou_cpu.cpp',
                    'src/beviou_api.cpp',
                ]
            ),
        ],
    )
