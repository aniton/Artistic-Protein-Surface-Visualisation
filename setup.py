import os
import sysconfig
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

name = 'surface'

class WithExternal(build_ext):
    def run(self):
        os.system(f"pip install github-clone")  # for cloning a specific directory of repo    
        os.system(f"wget 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'")
        os.system(f"wget 'https://s3-us-west-2.amazonaws.com/wengaoye/vgg19_normalised.npz'") # for AdaIN
        os.system(f"git clone 'https://github.com/danielgatis/rembg.git'") # for removing background
        os.system(f"git clone 'https://github.com/aniton/3D_PRoteins_Params.git'") # for 3D PyMOL Stylization
        os.replace('./2d_cnn/bg.py','./rembg/src/rembg/bg.py')
        os.chdir('./3d')
        os.system(f"ghclone 'https://github.com/hiroharu-kato/neural_renderer/tree/master/neural_renderer'") # for 3d style transfer
        
        os.chdir('../2d_cyclegan') # for CycleGan model:
        os.system(f"ghclone 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/data'")
        os.system(f"ghclone 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/models'")
        os.system(f"ghclone 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options'")
        os.system(f"ghclone 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/util'")
        os.replace('util.py','./util/util.py')
        os.mkdir('./datasets')
        os.system(f"install -D '../example/style_black.png' './datasets/trainB/style_black.png'") # fixed style image
        os.system(f"apt install imagemagick") # for making gifs
        build_ext.run(self)
        

setup(
    name=name,
    version='0.0.1',
    packages=find_packages(include=(name,)),
    cmdclass={'build_ext': WithExternal},
    descriprion='Repository of the industrial immersion project on Artistic Protein Surface Visualisation at Skoltech (summer 2021)',
    install_requires=requirements
)
