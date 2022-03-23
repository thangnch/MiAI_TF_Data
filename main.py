import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_path = 'cat_dog/'

datagen = ImageDataGenerator(rescale=1.0/255.0)

train_data = datagen.flow_from_directory(data_path + "train/",
  class_mode='binary', batch_size=64, target_size=(224, 224))


def check(dataset, n):
	# start our timer
	start = time.time()
	# loop over the provided number of steps
	for i in range(0, n):
		# get the next batch of data (we don't do anything with the
		# data since we are just benchmarking)
		(_, _) = next(dataset)
	return time.time() - start

print(check(train_data,500))
#103.76315593719482
