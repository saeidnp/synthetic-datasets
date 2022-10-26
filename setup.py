from distutils.core import setup
from setuptools import find_packages

setup(
    name='synthetic-datasets',
    version='0.0.1',
    py_modules=["synthetic_datasets"],
    license='MIT License',
    install_requires=[
            'sklearn',
            'matplotlib',
            'numpy',
            'common_utils @ git+ssh://git@github.com/saeidnp/common-utils.git@master#egg=common_utils',
            # 'torch>=1.0.1',
            # 'torchvision>=0.2.2'
        ],
)