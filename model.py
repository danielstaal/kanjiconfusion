import matplotlib.pyplot as plt
import numpy as np
import timeit
from random import shuffle
import sys
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()

import tiff2dataset
dataset = tiff2dataset.get_dataset()

# function to get a part of the dataset
def get_datapairs(dataset, n1, n2):
	return (dataset[0][n1:n2], dataset[1][n1:n2])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def learn():
	no_of_classes = len(dataset[1][0])
	print("no. of classes: ", no_of_classes)
	print("dataset size: ", len(dataset[0]))
	len_imagevector = len(dataset[0][0])
	train_part = int(len(dataset[0]) * 0.6666)

	# random permutation of dataset
	shuffle(dataset[0])
	shuffle(dataset[1])
	# then split the data in train and testset
	train_set = get_datapairs(dataset, 0, train_part)
	test_set = get_datapairs(dataset, train_part+1, len(dataset[0]))

	x = tf.placeholder(tf.float32, shape=[None, len_imagevector])
	y_ = tf.placeholder(tf.float32, shape=[None, no_of_classes])

	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1,48,48,1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([12 * 12 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, no_of_classes])
	b_fc2 = bias_variable([no_of_classes])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess.run(tf.global_variables_initializer())

	print("Training Phase")
	no_of_epoch = 200
	for i in range(1, no_of_epoch + 1):
		sys.stdout.write("\rEpoch: %i" % i)
		sys.stdout.flush()
		batch = train_set
		if i%10 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
			print(" training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
	print('')

	testimages = test_set[0]
	testlabels = test_set[1]
	print("Test Phase")
	print('accuracy: ', accuracy.eval(feed_dict={x: testimages, y_: testlabels, keep_prob: 1.0}))

if __name__ == '__main__':
	# start = timeit.timeit()
	
	learn()

	# end = timeit.timeit()
	# print("time taken: ", end - start)