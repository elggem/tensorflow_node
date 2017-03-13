from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['tensorflow_node', 'infogan'],
    package_dir={'': 'src', 'infogan': 'src/InfoGAN/infogan'}
)

setup(**d)
