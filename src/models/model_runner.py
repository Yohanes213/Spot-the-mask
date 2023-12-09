import pandas as pd
import os
import json
from data_prep import *

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator,


#
if __name__ == '__main__':



	current_directory = os.path.dirname(os.path.abspath(__file__))

	# Path to the train_labels.csv file

	csv_file_path = os.path.join(current_directory, os.pardir, os.pardir, 'data', 'processed', 'train_labels.csv')

	# Path to the config.json file

	json_file_path = os.path.join(current_directory, os.pardir, os.pardir, 'config.json')

	# Path to the image file

	image_path = os.path.join(current_directory, os.pardir, os.pardir, 'data', 'processed', 'images/')



	config = config_reader(json_file_path)

	callbacks = [
        EarlyStopping(**config["early_stopping"]),
        ReduceLROnPlateau(**config["reduce_lr_on_plateau"]),
        ModelCheckpoint(**config["model_checkpoint"]),
    ]



	df = csv_reader(csv_file_path)
	datalist = df.values
	image_width = config['image_width']
	image_height = config['image_height']
	dropout = config['dropout']

	train_name, train_img, train_label = prepare_data(datalist,image_path,image_width, image_height)

	train_name = np.asarray(train_name)
	train_img = np.asarray(train_img)
	train_label = np.asarray(train_label)
	train_img = train_img/255.0

	X_train, X_test, y_train, y_test = split_data(train_img, train_label, test_size=0.2)


	optimizer_config = config["optimizer"]
	optimizer_name = optimizer_config["name"]
	optimizer_amsgrad = optimizer_config["amsgrad"]
	optimizer = getattr(tf.keras.optimizers, optimizer_name)(amsgrad=optimizer_amsgrad)

	aug_config = config["data_augmentation"]
	aug = ImageDataGenerator(**aug_config)

	model = create_model(X_train, dropout,  optimizer)

	epochs = config["epochs"]

	history = train_model(model, aug, callbacks, epochs, X_train, y_train, batch)





