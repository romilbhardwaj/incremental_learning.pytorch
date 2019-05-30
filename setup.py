from setuptools import setup, find_packages

setup(
    name='inclearn',
    version='0.1a',
    url='https://github.com/arthurdouillard/incremental_learning.pytorch',
    author='arthurdouillard',
    description='Incremental learning',
    packages=find_packages(),
    install_requires=['torch'],
)