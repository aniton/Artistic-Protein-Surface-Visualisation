import os
import sysconfig
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext



with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

name = 'surface'

class WithExternal(build_ext):
    def run(self):
        os.system(f"pip install git+git://github.com/HR/github-clone#egg=ghclone")  # for cloning a specific directory of repo    
        os.system(f"wget 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'")
        os.system(f"apt install imagemagick") # for making gifs
        os.chdir('./3d')
        os.system(f"ghclone 'https://github.com/hiroharu-kato/neural_renderer/tree/master/neural_renderer'") # for 3d style transfer
	"""
	For CycleGan model:
	"""
	os.chdir('../2d_cyclegan')
	os.system(f"ghclone 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/data'")
	os.system(f"ghclone 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/models'")
	os.system(f"ghclone 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options'")
	os.system(f"ghclone 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/util'")
	os.mkdir('./datasets')
	os.system(f"cp '../example/style_black.png ./datasets/trainB'") # fixed style image

        build_ext.run(self)

setup(
    name=name,
    version='0.0.1',
    packages=find_packages(include=(name,)),
    cmdclass={'build_ext': WithExternal},
    descriprion='Repository of the industrial immersion project on Artistic Protein Surface Visualisation at Skoltech (summer 2021)',
    install_requires=requirements
)