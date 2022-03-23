import tensorflow as tf
from imutils import paths
import numpy as np
import os
import time

data_path = 'cat_dog/'


train_images_path = list(paths.list_images(data_path + "train/"))
classNames = np.array(sorted(os.listdir(data_path + "train/")))

def load_images(imagePath):
	# read the image from disk, decode it, resize it, and scale the
	# pixels intensities to the range [0, 1]
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_png(image, channels=3)
	image = tf.image.resize(image, (224, 224)) / 255.0
	# grab the label and encode it
	label = tf.strings.split(imagePath, os.path.sep)[-2]
	oneHot = label == classNames
	encodedLabel = tf.argmax(oneHot)
	# return the image and the integer encoded label
	return (image, encodedLabel)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images_path)
train_dataset = (train_dataset
	.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
	.batch(64)
	.repeat()
	.prefetch(tf.data.AUTOTUNE)
)

def check(dataset, n):
	# start our timer
	start = time.time()
	# loop over the provided number of steps
	for i in range(0, n):
		# get the next batch of data (we don't do anything with the
		# data since we are just benchmarking)
		(_, _) = next(dataset)
	return time.time() - start
train_dataset_iter = iter(train_dataset)
print(check(train_dataset_iter,500))
#43.47282814979553

