import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_version(version_file):
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


os.chdir(osp.dirname(osp.abspath(__file__)))
csrc_directory = osp.join('IF4DLUT', 'csrc')
setup(
    name='IF4DLUT',
    version=get_version(osp.join('IF4DLUT', 'version.py')),
    description='LookUp Table Transform',
    author='ZN Cai',
    packages=find_packages(),
    include_package_data=False,
    ext_modules=[
        CUDAExtension('IF4DLUT._ext', [
            osp.join(csrc_directory, 'IF4DLUT_transform.cpp'),
            osp.join(csrc_directory, 'IF4DLUT_transform_cuda.cu')
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    license='Apache License 2.0',
    zip_safe=False
)
