import tensorflow as tf
from model import *
from datetime import datetime
import numpy as np
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')
MODEL_SAVE_PATHS = './models/'
EPOCHS = 20
EPSILON = 1e-5
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
HEIGHT=256
WIDTH=256
CHANNEL=3
INPUT_SHAPE = (BATCH_SIZE,HEIGHT, WIDTH, CHANNEL)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vgg_path', dest='vgg_path', type=str, default='vgg19_normalised.npz')
    parser.add_argument('--content_path', dest='content_path', type=str, default='./proteins')
    parser.add_argument('--style_path', dest='style_path', type=str, default='./goodsell')

    args = parser.parse_args()

    start_time = datetime.now()
    content_images = list_images(args.content_path)
    style_images = list_images(args.style_path)
    num_imgs = min(len(content_images), len(style_images))
    content_images = content_images[:num_imgs]
    style_images = style_images[:num_imgs]
    mod = num_imgs % BATCH_SIZE
    if mod > 0:
        content_images = content_images[:-mod]
        style_images = style_images[:-mod]

    with tf.Graph().as_default(), tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:

        encoder = Encoder(args.vgg_path)
        decoder = Decoder()

        content_input = tf.compat.v1.placeholder(tf.float32, shape=INPUT_SHAPE, name='content_input')
        style_input = tf.compat.v1.placeholder(tf.float32, shape=INPUT_SHAPE, name='style_input')
        content = tf.reverse(content_input, axis=[-1])
        style = tf.reverse(style_input, axis=[-1])
        content = encoder.preprocess(content)
        style =  encoder.preprocess(style)

        content_features,content_layers = encoder.encode(content)
        style_features,style_layers = encoder.encode(style)

        adain_features = AdaIN(content_features,style_features)


        stylied_image = decoder.decode(adain_features)

        stylied_image = encoder.deprocess(stylied_image)

        stylied_image = tf.reverse(stylied_image, axis=[-1])

        stylied_image = tf.clip_by_value(stylied_image, 0.0, 255.0)
        stylied_image = tf.reverse(stylied_image, axis=[-1])
        stylied_image = encoder.preprocess(stylied_image)
        stylied_features,stylied_layers = encoder.encode(stylied_image)
        content_loss = tf.reduce_sum(tf.reduce_mean(tf.square(stylied_features - adain_features), axis=[1, 2]))
        style_layer_loss = []
        for layer in STYLE_LAYERS:
            enc_style_feat   = style_layers[layer]
            enc__stylied_feat  = stylied_layers[layer]

            meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
            meanG, varG = tf.nn.moments(enc__stylied_feat,   [1, 2])

            sigmaS = tf.sqrt(varS + EPSILON)
            sigmaG = tf.sqrt(varG + EPSILON)

            l2_mean  = tf.reduce_sum(tf.square(meanG - meanS))
            l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

            style_layer_loss.append(l2_mean + l2_sigma)

        style_loss = tf.reduce_sum(style_layer_loss)

        style_weight = 2.0
        loss = content_loss + style_weight * style_loss
        train_op = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        step = 0
        n_batches = int(num_imgs// BATCH_SIZE)

        elapsed_time = datetime.now() - start_time
        print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)

        print('Now begin to train the model...\n')
        start_time = datetime.now()

        for epoch in range(EPOCHS):

            np.random.shuffle(content_images)
            np.random.shuffle(style_images)

            for batch in range(n_batches):
                content_batch_path = content_images[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                style_batch_path = style_images[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]

                content_batch = get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
                style_batch = get_train_images(style_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
                sess.run(train_op, feed_dict={content_input: content_batch, style_input: style_batch})

                if step % 100 == 0:

                    _content_loss, _style_loss, _loss = sess.run([content_loss, style_loss, loss],
                                                                 feed_dict={content_input: content_batch,
                                                                            style_input: style_batch})

                    elapsed_time = datetime.now() - start_time
                    print('step: %d,  total loss: %.3f, elapsed time: %s' % (step, _loss,elapsed_time))
                    print('content loss: %.3f' % (_content_loss))
                    print('style loss  : %.3f,  weighted style loss: %.3f\n' % (
                    _style_loss, style_weight * _style_loss))

                if step % 1000 == 0:
                    print('save model now,step:',step)
                    saver.save(sess, MODEL_SAVE_PATHS, global_step=step)

                step += 1
            print(f'{epoch + 1} Epoch finished\n!')
        saver.save(sess, MODEL_SAVE_PATHS)