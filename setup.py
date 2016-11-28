from setuptools import find_packages
from setuptools import setup

setup(name='destin',
      version='0.1',
      description='Tensorflow implementation of DeSTIN',
      url='http://github.com/elggem/ae-destin',
      author='Ralf Mayet',
      author_email='mail@elggem.pub',
      license='Unlicense',
      packages=find_packages(),
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False)
