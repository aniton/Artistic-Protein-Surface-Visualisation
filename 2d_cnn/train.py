"""
Based on Fast Style Transfer https://github.com/lengstrom/fast-style-transfer.git
"""
from __future__ import print_function
import sys, os, pdb
import numpy as np, scipy.misc 
import optimize
import model
from argparse import ArgumentParser
import scipy.misc
import tensorflow as tf
from collections import defaultdict
import time
import json
from tqdm import tqdm
import subprocess
import numpy
import utils

DEVICE = '/gpu:0'
FRAC_GPU = 1

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--run_path', type=str,
                        dest='run_path',
                        help='path to .mat weights',
                         default='./imagenet-vgg-verydeep-19.mat')
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                         default='./2d_cnn/checkpoints')

    parser.add_argument('--style_dir', type=str,
                        dest='style_dir', help='style images path',
                         required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                         default='./data_generation/train_pdb')

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                         default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs', default=100)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size', default=4)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        default=2000)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight',
                        default=7.5e0)
    
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight',
                        default=1e2)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight', default=2e2)

    parser.add_argument('--shift', type=float,
                        dest='shift',
                        help='with (1) or w/o (0) shifted activations when computing Gram matrices',
                        default=0)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=1e-3)

    return parser

def check_opts(opts):
    utils.exists(opts.checkpoint_dir, "checkpoint dir not found!")
    utils.exists(opts.style_dir, "style path not found!")
    utils.exists(opts.train_path, "train path not found!")
    utils.exists(opts.run_path, "Model weights not found!")
    if opts.test or opts.test_dir:
        utils.exists(opts.test, "test img not found!")
        utils.exists(opts.test_dir, "test directory not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.run_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0

def _get_files(img_dir):
    files = utils.list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]


def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = utils.get_img(data_in[0]).shape
    else:
        assert data_in.size[0] == len(paths_out)
        img_shape = X[0].shape

    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    curr_num = 0
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.compat.v1.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = model.net(img_placeholder)
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = utils.get_img(path_in)
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            for j, path_out in enumerate(curr_batch_out):
                utils.save_img(path_out, _preds[j])
                
        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]
    if len(remaining_in) > 0:
        ffwd(remaining_in, remaining_out, checkpoint_dir, 
            device_t=device_t, batch_size=1)

def verify(dirname):
      if not os.path.exists(dirname):
          os.mkdir(dirname)
          print("Directory " , dirname,  " Created ")
      else:    
          print("Directory " , dirname,  " already exists")

def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0'):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device)

def run():
    parser = build_parser()
    options = parser.parse_args()
    verify(options.checkpoint_dir)
    verify(options.test_dir)
    check_opts(options)
    style_targets = _get_files(options.style_dir)
    content_targets = _get_files(options.train_path)

    for style in style_targets:
        style_target = utils.get_img(style)
        kwargs = {
          "epochs":options.epochs,
          "print_iterations":options.checkpoint_iterations,
          "batch_size":options.batch_size,
          "save_path":os.path.join(options.checkpoint_dir,'fns.ckpt'),
          "learning_rate":options.learning_rate
         }
        args = [
         options.run_path,
         content_targets,
         style_target,
         options.content_weight,
         options.style_weight,
         options.tv_weight,
         options.shift
        ]

        for preds, losses, i, epoch in tqdm(optimize.optimize(*args, **kwargs)):
          style_loss, content_loss, tv_loss, loss = losses

          print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
          to_print = (style_loss, content_loss, tv_loss)
          print('style: %s, content:%s, tv: %s' % to_print)
          if options.test:
              assert options.test_dir != False
              preds_path = '%s/%s_%s_%s.png' % (options.test_dir,epoch,i, os.path.basename(style))          
              ckpt_dir = os.path.dirname(options.checkpoint_dir)
              ffwd_to_img(options.test,preds_path,
                                     options.checkpoint_dir)

        ckpt_dir = options.checkpoint_dir
        print("Training complete")

if __name__ == '__main__':
    run()
