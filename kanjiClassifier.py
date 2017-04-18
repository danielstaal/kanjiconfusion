import tensorflow as tf
import numpy as np


def create_graph(params):

    input_dim = params['input_dim']
    n_class = params['n_class']
    n_inp_channel = params['n_inp_channel']
    filter_dim = params['filter_dim']
    n_filter = params['n_filter']
    conv_strides = params['conv_strides']
    n_layer = len(filter_dim)
    learning_rate = params['learning_rate']
    n_filter = [n_inp_channel] + n_filter
    l2_beta = params['l2_beta']


    # placeholder for dropout
    keep_prob = tf.placeholder(tf.float32)

    # first conv layer
    x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim, input_dim, n_inp_channel])
    inp_img = x
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_class])
    for l in range(n_layer):
        w = tf.Variable(tf.truncated_normal([filter_dim[l], filter_dim[l], n_filter[l], n_filter[l+1]],
                                             stddev=0.1), dtype=tf.float32, name = 'v_' + str(n_layer))
        # b1 = tf.Variable(tf.truncated_normal([]), name='b1')
        conv_l = tf.nn.conv2d(x, w, strides=[1, conv_strides[l], conv_strides[l], 1], padding='SAME')

        # dropout
        xy = tf.nn.relu(conv_l)
        x = tf.nn.dropout(xy, keep_prob)


    # flatten the last convulation layer and add the output layer
    last_conv = tf.contrib.layers.flatten(x)
    n_neuron_last_layer = last_conv.get_shape().as_list()[-1]
    w_out = tf.Variable(tf.truncated_normal([n_neuron_last_layer, n_class], stddev=0.1), dtype=tf.float32, name='v_out')
    y_hat = tf.matmul(last_conv, w_out)

    # make loss function
    # l2 reg
    trainable_vars = tf.trainable_variables()
    l2_reg = tf.nn.l2_loss([tf.nn.l2_loss(v) for v in trainable_vars]) * l2_beta
    # loss function
    loss = tf.add(tf.reduce_sum(tf.squared_difference(y_hat, y)), l2_reg)

    # optimizer step
    step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return inp_img, y, y_hat, loss, step


def main(params):
    n_epoch = params['n_epoch']
    batch_size = params['batch_size']
    inp_img, y, y_hat, loss, step = create_graph(params)
    destination_location = params['destination_location']
    dropout_val = params['dropout_prob']
    outputfile = params['outputfile']

    # initialize graph
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)





if __name__ == '__main__':
    params = dict()
    params['input_dim'] = 48
    params['n_class'] = 100
    params['n_inp_channel'] = 3
    params['n_filter'] = [64, 64, 64, 64, 64]
    params['filter_dim'] = [5, 3, 5, 3, 3]
    params['conv_strides'] = [1, 2, 1, 2, 1]
    params['destination_location'] = 'latest_model/'
    params['l2_beta'] = 0.02
    params['n_epoch'] = 20000 # 40000
    params['batch_size'] = 128
    params['learning_rate'] = 0.001
    params['dropout_prob'] = 0.6
    params['outputfile'] = '128_128_point2.ckpt'

    main(params)
