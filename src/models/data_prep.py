import json
import pandas as pd
import glob
import os

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img,array_to_img, img_to_array
from keras.layers import Dropout, Dense,GlobalAveragePooling2D
import efficientnet.keras as efn
from keras.models import Model


def config_reader(path: str) -> dict:
    """
    Reads and parses a JSON configuration file.

    Args:
        path: The path to the JSON configuration file.

    Returns:
        A dictionary containing the parsed configuration data.
    """

    with open(path, "r") as config_file:
        # Load the JSON data from the file.
        try:
            config = json.load(config_file)
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing JSON config file: {e}")

    # Validate the config data type.
    if not isinstance(config, dict):
        raise Exception("Config data must be a dictionary.")

    return config



def csv_reader(path: str) -> pd.DataFrame:
  """Reads a CSV file and returns a pandas DataFrame."""
  return pd.read_csv(path)



def prepare_data(datalist, image_path, image_width, image_height):
    """
    Prepares data for training.

    Args:
        datalist: A list of tuples containing filenames and labels.
        image_path: The path to the image directory.
        image_width: The desired image width.
        image_height: The desired image height.

    Returns:
        A tuple containing two lists: (names, images, labels).
    """

    names = []
    images = []
    labels = []

    for filename, label in datalist:
        # Construct the full path to the image.
        full_path = os.path.join(image_path, filename)

        # Load the image.
        image = load_img(full_path, target_size=(image_width, image_height))

        # Convert the image to a NumPy array.
        image = img_to_array(image, color_mode="rgb")

        # Append the data to the corresponding lists.
        names.append(filename)
        images.append(image)
        labels.append(label)

    return names, images, labels



def split_data(train_img, train_label, test_size=0.2):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(train_img, train_label, test_size=test_size)
    return X_train, X_test, y_train, y_test




def create_model(X_train ,  dropout, optimizer):
    """
    Creates a deep learning model for image classification.

    Args:
        X_train: A NumPy array containing the training images.

    Returns:
        A compiled Keras model for image classification.
    """

    # Load the EfficientNetB0 pre-trained model with ImageNet weights.
    # We exclude the top fully-connected layers as we will be replacing them.
    base_model = efn.EfficientNetB0(weights="imagenet", include_top=False, input_shape=X_train[0].shape)

    # Access the last output layer of the pre-trained model.
    last_output = base_model.output

    # Apply global average pooling to the features extracted from the pre-trained model.
    x = GlobalAveragePooling2D()(last_output)

    # Apply batch normalization to stabilize the network.

    # Add a dropout layer with a rate of 0.2 to prevent overfitting.
    x = Dropout(dropout)(x)

    # Add a dense layer with 1 output neuron and sigmoid activation for binary classification.
    x = Dense(1, activation="sigmoid")(x)

    # Create a new Keras model with the pre-trained model as input and the new output layer.
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model with the Adam optimizer and binary cross-entropy loss function.
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model



def train_model(model, aug, callbacks, epochs, X_train, y_train, batch):
    """
    Trains the provided model with data augmentation and specified callbacks.

    Args:
        model: The compiled Keras model to be trained.
        aug: The data augmentation generator.
        callbacks: A list of Keras callbacks to monitor and control training.
        epochs: The number of training epochs.
        X_train: The NumPy array of training images.
        y_train: The NumPy array of training labels.
        batch: The training batch size.

    Returns:
        None
    """

    # Train the model using the data augmentation generator and specified callbacks.
    model.fit_generator(
        generator=aug.flow(X_train, y_train, batch_size=batch),
        validation_data=(X_test, y_test),  # Evaluate on the validation set after each epoch
        epochs=epochs,  # Train for the specified number of epochs
        callbacks=callbacks  # Use the provided callbacks to monitor and control training
    )