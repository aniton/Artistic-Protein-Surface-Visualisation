from __future__ import print_function
import functools
import pdb, time
import tensorflow as tf, numpy as np, os
import model
import scipy.io as sio
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

def preprocess(image):
    return image - np.array([ 123.68 ,  116.779,  103.939])

def loss_vgg_model(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = sio.loadmat(data_path)
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = conv2d(current, kernels, bias)
            current = tf.nn.relu(current)
        elif kind == 'relu':
            current = relu(current)
        elif kind == 'pool':
            current = pool(current)
        
        net[name] = current

    assert len(net) == len(layers)
    return net

def conv2d(input, weights, bias):
    conv = tf.nn.conv2d(input=input, filters=tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)

def pool(input):
    return tf.nn.max_pool2d(input=input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def relu(input):
    return tf.nn.relu(input)

def optimize(run_path, content_targets, style_target, content_weight, style_weight,
             tv_weight, shift, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt',
             learning_rate=1e-3,  debug=False):

    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 

    style_features = {}

    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape
    print(style_shape)

    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.compat.v1.Session() as sess:
        style_image = tf.compat.v1.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = preprocess(style_image)
        net = loss_vgg_model(run_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T - shift, features - shift) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        X_content = tf.compat.v1.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = loss_vgg_model(run_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        preds = model.net(X_content/255.0)
        preds_pre = preprocess(preds)

        net = loss_vgg_model(run_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(a=feats, perm=[0,2,1])
            grams = tf.matmul(feats_T - shift, feats - shift) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        loss = content_loss + style_loss + tv_loss

        # overall loss
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.compat.v1.global_variables_initializer())
        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch
                }

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch
                    }

                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    saver = tf.compat.v1.train.Saver()
                    res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)
