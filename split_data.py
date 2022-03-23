from shutil import copyfile
import os
from random import seed
from random import random
# create directories

data_path = 'cat_dog/'

subdirs = ['train/', 'val/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['dogs/', 'cats/']
	for labldir in labeldirs:
		newdir = data_path + subdir + labldir
		os.makedirs(newdir, exist_ok=True)

# seed random number generator
seed(42)
# define ratio of pictures to use for validation
val_ratio = 0.2
# copy training dataset images into subdirectories
for file in os.listdir(data_path):
	src = data_path + '/' + file
	dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'val/'
	if file.startswith('cat'):
		dst = data_path + dst_dir + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = data_path + dst_dir + 'dogs/'  + file
		copyfile(src, dst)