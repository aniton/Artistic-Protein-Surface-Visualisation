import os
import sysconfig
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext



with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

name = 'surface'


setup(
    name=name,
    version='0.0.1',
    packages=find_packages(include=(name,)),
    descriprion='Repository of the industrial immersion project on Artistic Protein Surface Visualisation at Skoltech (summer 2021)',
    install_requires=requirements
)