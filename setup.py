import os
import sysconfig
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext



with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

name = 'surface'

class WithExternal(build_ext):
    def run(self):
        os.chdir(os.getenv("HOME"))
        os.system(f'wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat')
        build_ext.run(self)

setup(
    name=name,
    version='0.0.1',
    packages=find_packages(include=(name,)),
    cmdclass={'build_ext': WithExternal},
    descriprion='Repository of the industrial immersion project on Artistic Protein Surface Visualisation at Skoltech (summer 2021)',
    install_requires=requirements
)