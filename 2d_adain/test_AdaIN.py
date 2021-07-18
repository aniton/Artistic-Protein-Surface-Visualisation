import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave, imresize
import os
import argparse
import imageio
from model import *
import sys
from PIL import ImageFile

sys.path.append('./rembg/src/rembg/')
from bg import *
output_path = './output/'
ImageFile.LOAD_TRUNCATED_IMAGES = True

def verify(dirname):
      if not os.path.exists(dirname):
          os.mkdir(dirname)
          print("Directory " , dirname,  " Created ")
      else:    
          print("Directory " , dirname,  " already exists")
def remove_back(out_path):
    f = np.fromfile(out_path)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(out_path, format='png') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vgg_path', dest='vgg_path', type=str, default='vgg19_normalised.npz')
    parser.add_argument('--content_path', dest='content_path', type=str, default='./proteins')
    parser.add_argument('--style_path', dest='style_path', type=str, default='./goodsell')
    parser.add_argument('--weights', dest='weights', type=str, default='weights.npy')

    args = parser.parse_args()
    content_images = os.listdir(args.content_path)
    style_images = os.listdir(args.style_path)
    verify(output_path)
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:

        encoder = Encoder(args.vgg_path)
        decoder = Decoder(mode='test', weights_path=args.weights)

        content_input = tf.compat.v1.placeholder(tf.float32, shape=(1,None,None,3), name='content_input')
        style_input =   tf.compat.v1.placeholder(tf.float32, shape=(1,None,None,3), name='style_input')

        content = tf.reverse(content_input, axis=[-1])
        style   = tf.reverse(style_input, axis=[-1])
        content = encoder.preprocess(content)
        style   = encoder.preprocess(style)

        enc_c, enc_c_layers = encoder.encode(content)
        enc_s, enc_s_layers = encoder.encode(style)
        target_features = AdaIN(enc_c, enc_s)
        generated_img = decoder.decode(target_features)
        generated_img = encoder.deprocess(generated_img)
        generated_img = tf.reverse(generated_img, axis=[-1])
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)
        sess.run(tf.compat.v1.global_variables_initializer())

        for s in style_images:
            for c in content_images:
                content_image = imageio.imread(os.path.join(args.content_path,c), pilmode='RGB')
                style_image   = imageio.imread(os.path.join(args.style_path,s), pilmode='RGB')
                content_tensor = np.expand_dims(content_image, axis=0)
                style_tensor = np.expand_dims(style_image, axis=0)

                result = sess.run(generated_img, feed_dict={content_input: content_tensor,style_input: style_tensor})
                result_name = os.path.join(output_path,s.split('.')[0]+'_'+c.split('.')[0]+'.jpg')
                print(result_name,' is generated')
                imageio.imwrite(result_name, result[0])
                remove_back(result_name)
