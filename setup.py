# -*- coding: utf-8 -*-

from setuptools import setup
from gpu_control.gpu_manager import __version__, __author__

setup(name='gpu_control',
      version=__version__,
      description='gpu_control.py: for choosing and setting gpus to run framework',
      author=__author__,
      maintainer=__author__,
      url='https://github.com/DogfishBone/GpuManager',
      packages=['gpu_control'],
      long_description="Make setting gpus more easy.",
      license="Public",
      platforms=["any"],
)
