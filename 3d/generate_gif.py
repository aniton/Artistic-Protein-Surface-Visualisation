"""
Based on Neural 3D Mesh Renderer https://github.com/hiroharu-kato/neural_renderer
"""
import argparse
import sys
import glob
import os
import subprocess
import imageio 
import chainer
import chainer.functions as cf
import numpy as np
import scipy.misc
sys.path.append('./rembg/src/rembg/')
from bg import *
import tqdm
sys.path.append('./3d/neural_renderer/')
import neural_renderer


class Model(chainer.Link):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        with self.init_scope():
            # load .obj
            vertices, faces = neural_renderer.load_obj(filename_obj)
            self.vertices = vertices[None, :, :]
            self.faces = faces[None, :, :]

            # create textures
            texture_size = 6
            textures = np.zeros((1, self.faces.shape[1], texture_size, texture_size, texture_size, 3), 'float32')
            self.textures = chainer.Parameter(textures)

            # load reference image
            self.image_ref = imageio.imread(filename_ref).astype('float32') / 255.

            # setup renderer
            renderer = neural_renderer.Renderer()
            renderer.perspective = False
            renderer.light_intensity_directional = 0.0
            renderer.light_intensity_ambient = 1.0
            self.renderer = renderer

    def to_gpu(self, device=None):
        super(Model, self).to_gpu(device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)
        self.vertices = chainer.cuda.to_gpu(self.vertices, device)
        self.image_ref = chainer.cuda.to_gpu(self.image_ref, device)

    def __call__(self):
        self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        image = self.renderer.render(self.vertices, self.faces, cf.tanh(self.textures))
        loss = cf.sum(cf.square(image - self.image_ref.transpose((2, 0, 1))[None, :, :, :]))
        return loss


def make_gif(working_directory, filename):
    options = '-delay 7 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, filename), shell=True)
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)

def remove_back(out_path):
    f = np.fromfile(out_path)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(out_path)
    
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default='./example/4L6R.obj')
    parser.add_argument('-ir', '--filename_ref', type=str, default='./example/style_small.jpg')
    parser.add_argument('-is', '--num_opt_steps', type=str, default=300)
    parser.add_argument('-or', '--filename_output', type=str, default=f"./result.gif")
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output)

    model = Model(args.filename_obj, args.filename_ref)
    model.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=0.1, beta1=0.5)
    optimizer.setup(model)
    loop = tqdm.tqdm(range(int(args.num_opt_steps)))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.target.cleargrads()
        loss = model()
        loss.backward()
        optimizer.update()

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, azimuth)
        images = model.renderer.render(model.vertices, model.faces, cf.tanh(model.textures))
        image = images.data.get()[0].transpose((1, 2, 0))
        if num == 0:
            scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/mesh22d.png' % ('./example')) # achieve 2d representation
            remove_back('%s/mesh22d.png' % ('./example'))
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, num))
    make_gif(working_directory, args.filename_output)


if __name__ == '__main__':
    run()
