import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def compute_accuracy(y_hat_val, batch_label):
    pred_cls = np.argmax(y_hat_val, axis=1)
    act_cls = np.argmax(batch_label, axis=1)
    acc = np.sum(pred_cls == act_cls)

    return acc


def get_batch(tr_image, tr_labels, batch_size):
    seek_sample = get_batch.seek  # current position of the seek pointer
    n_sample = tr_image.shape[0]

    # select samples based on the current seek pointer and the batch_size
    sel_samples = np.arange(seek_sample, seek_sample + batch_size)
    # reset to index zero when end of data set is reached
    sel_samples[sel_samples >= n_sample] -= n_sample
    batch = tr_image[sel_samples, :, :, :]  # create the batch
    batch_label = tr_labels[sel_samples, :]

    # update seek pointer
    get_batch.seek += batch_size
    if get_batch.seek >= n_sample:
        get_batch.seek -= n_sample

    return batch, batch_label


def get_batch_tst(tr_image, tr_labels, batch_size):
    seek_sample = get_batch_tst.seek  # current position of the seek pointer
    n_sample = tr_image.shape[0]

    # select samples based on the current seek pointer and the batch_size
    sel_samples = np.arange(seek_sample, seek_sample + batch_size)
    # reset to index zero when end of data set is reached
    sel_samples[sel_samples >= n_sample] -= n_sample
    batch = tr_image[sel_samples, :, :, :]  # create the batch
    batch_label = tr_labels[sel_samples, :]

    # update seek pointer
    get_batch_tst.seek += batch_size
    if get_batch_tst.seek >= n_sample:
        get_batch_tst.seek -= n_sample

    return batch, batch_label


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

    return inp_img, y, y_hat, loss, step, keep_prob


def main(params):
    n_epoch = params['n_epoch']
    batch_size = params['batch_size']
    inp_img, y, y_hat, loss, step, keep_prob = create_graph(params)
    destination_location = params['destination_location']
    outputfile = params['outputfile']
    keep_prob_val = params['keep_prob']


    # initialize graph
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # load img data
    images = np.load('img_data/images_sample.dat')
    labels = np.load('img_data/labels_sample.dat')
    images_trn = images[0:80]
    labels_trn = labels[0:80]
    images_tst = images[80:100]
    labels_tst = labels[80:100]

    # set the index of the first sample to be read from the data
    get_batch.seek = 0
    acc = 0




    tr_data, tr_labels = load_cifar(norm_image=True)
    tst_data, tst_labels = load_cifar_test(norm_image=True)
    acc = 0
    for e in range(n_epoch):

        if e % 100 == 0:
            print('Accuracy = ' + str(acc * 100 / 10000))
            acc = 0

        batch, batch_label = get_batch(tr_data, tr_labels, batch_size)

        # run one step of optimizer
        y_hat_val, loss_val, _ = sess.run([y_hat, loss, step], feed_dict={inp_img: batch,
                                                                          y: batch_label, keep_prob: dropout_val})

        # compute accuracy
        acc += compute_accuracy(y_hat_val, batch_label)

        print('Error in epoch ' + str(e) + ' = ' + str(np.sum(loss_val)))

        # test accuracy
        if e % 200 == 0:
            # test accuracy on separate batch
            get_batch_tst.seek = 0
            tst_acc = 0

            for f in range(100):
                batch, batch_label = get_batch_tst(tst_data, tst_labels, batch_size)

                # run one step of optimizer
                y_hat_val = sess.run(y_hat, feed_dict={inp_img: batch, y: batch_label, keep_prob: 1})

                # compute accuracy
                tst_acc += compute_accuracy(y_hat_val, batch_label)

            print('accuracy on tstbtch is ' + str(tst_acc * 100 / 10000))












    # train the network
    for i in range(n_epoch):

        # load data and labels

        # load batch images and labels into batch
        batch, batch_label = get_batch(images_trn, labels_trn, batch_size)

        # do a training step
        y_hat_val, loss_val, _ = sess.run([y_hat, loss, step], feed_dict={inp_img:batch, y:batch_label, keep_prob:keep_prob_val})

        # evaluate
        if i % 40 == 0:
            batch_tst, batch_label_tst = get_batch_tst(images_tst, labels_tst, len(labels_tst))
            y_hat_tst_val = sess.run(y_hat, feed_dict={inp_img:batch_tst, y:batch_label_tst, keep_prob:1.0})
            acc = compute_accuracy(y_hat_tst_val, batch_label_tst)
            print('accuracy in ' + str(i) + ' itterations is ' + str(acc) + ' this is on epoch ' + str(loss_val))


if __name__ == '__main__':
    params = dict()
    params['input_dim'] = 48
    params['n_class'] = 40
    params['n_inp_channel'] = 1
    params['n_filter'] = [16, 16, 16, 16]
    params['filter_dim'] = [5, 3, 5, 3]
    params['conv_strides'] = [1, 2, 1, 2]
    params['destination_location'] = 'latest_model/'
    params['l2_beta'] = 0.02
    params['n_epoch'] = 20000 # 40000
    params['batch_size'] = 10
    params['learning_rate'] = 0.01
    params['outputfile'] = '128_128_point2.ckpt'
    params['keep_prob'] = 1

    main(params)
