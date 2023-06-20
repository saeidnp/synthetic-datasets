from distutils.core import setup
from setuptools import find_packages

setup(
    name='synthetic-datasets',
    version='0.0.1',
    packages=["synthetic_datasets"],
    license='MIT License',
    install_requires=[
            'scikit-learn',
            'matplotlib',
            'numpy',
            'common_utils @ git+ssh://git@github.com/saeidnp/common-utils.git@master#egg=common_utils',
        ],
)
