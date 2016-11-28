from setuptools import setup

setup(name='destin',
      version='0.2',
      description='Tensorflow implementation of DeSTIN',
      url='http://github.com/elggem/ae-destin',
      author='Ralf Mayet',
      author_email='mail@elggem.pub',
      license='Unlicense',
      packages=["destin", "destin.utils", "destin.nodes", "destin.input"],
      install_requires=["colorlog"],
      zip_safe=False)
