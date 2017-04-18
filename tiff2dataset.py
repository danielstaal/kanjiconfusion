# import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
import codecs
import csv
import glob
import regex as re
np.set_printoptions(threshold=np.nan)
from PIL import Image

########## to time a function
# start = timeit.timeit()
# end = timeit.timeit()
# print(end - start)

########## To convert to PNG 8-bit:
# im = (255.0 / im.max() * (im - im.min())).astype(np.uint8)
# im = Image.fromarray(im)
# im.save('test.png')

########## To show the image
# im = plt.imshow(im, aspect="auto", cmap='gray')
# plt.show()

# pass all labels to create onehot dict
def create_onehot_dict(labels, onehot_dict):
	for label in labels:
		if label not in onehot_dict:
			onehot_dict[label] = len(onehot_dict)

# create datapoint by transforming the image(nparray) matrix into
# a vector and get the onehot vector for this label
def create_datapoint(im, label, onehot_dict):

	# change to vector
	im = np.asarray(im).reshape(-1)

	index = onehot_dict[label]
	one_hot_label = np.zeros(len(onehot_dict))
	one_hot_label[index] = 1
	
	return [im, one_hot_label]

# this function creates the dataset as a tuple of two lists:
# (image_vectors, image_labels)
def get_dataset():
	onehot_dict = {}

	all_labels = []

	label_file = codecs.open("output.csv", "r", "utf-8") 
	labels = label_file.readlines()

	for line in labels:
		writer_labels = str(line.encode('utf-8'))
		writer_labels = writer_labels[2:len(writer_labels)-5]
		writer_labels = writer_labels.split(',')

		all_labels.append(writer_labels)

	for writer in all_labels:
		create_onehot_dict(writer, onehot_dict)

	images = glob.glob('offline_competition_data/*.tiff')

	# initialize matrix with total amount of images rows, and amount of classes columns
	no_of_images = len(all_labels)*len(all_labels[0])
	no_of_classes = len(onehot_dict)
	image_vectors = np.zeros((no_of_images,48*48))
	onehot_labels = np.zeros((no_of_images,no_of_classes))

	for i in range(len(images)):
		s = images[i]
		writer_id = ''
		char_index = ''
		regex = re.compile(r'\d+')

		# dont put any other digits in the filename
		numbers = [int(x) for x in regex.findall(s)]
			
		writer_id = numbers[0]-1
		char_index = numbers[1]-1

		label = all_labels[writer_id][char_index]
		im = Image.open(images[i])
		im = np.array(im)

		[im, vec] = create_datapoint(im, label, onehot_dict)
		image_vectors[i] = im
		onehot_labels[i] = vec

	return (image_vectors, onehot_labels)

if __name__ == '__main__':
	i, j = get_dataset()
	print(i[1])