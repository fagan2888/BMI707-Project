# load_data.py
# A script containing functions for loading and augmenting data.
# Authors: Matthew West <mwest@hsph.harvard.edu>, Hillary Tsang, Anjali Jha

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image


def load_metadata(file_name='metadata.csv'):
    """A function to load and split metadata for COVID X-rays.
    
    Parameters
    ----------
    file_name : string
        String of metadata filename.
    
    Returns
    -------
    df_train : pandas DataFrame
        Dataframe of training metadata.

    df_val : pandas DataFrame
        Dataframe of validation metadata.
    """
 
    df = pd.read_csv(file_name)

    # Select views: AP/PA/AP Supine
    df = df[(df['view']=='PA') | (df['view']=='AP') 
              | (df['view']=='AP Supine') | (df['view']=='AP semi erect')]

    df_train, df_val = train_test_split(df, test_size=0.33, random_state=42)

    return df_train, df_val


def load_data(df, image_size=(256,256)):
    """A function to load and process images from metadata dataframes, into 
    NumPy arrays for use with Keras model.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame of metadata.

    image_size : tuple of int
        Size to resize images to.

    Returns
    -------
    images : array-like
        Array of grayscale NumPy mmages.
    
    labels : array-like
        List of labels.
    """    
    
    images, labels = [], []

    # Iterate over metadata
    for _, row in df.iterrows():
        # Get image path
        im_path = row['filename']
        
        # Open image from path
        im = Image.open('images/' + im_path)
        
        # Convert to grayscale
        im = im.convert('RGB')

        # Resize to square
        im = im.resize(image_size)
        
        # Convert to array, normalize and append
        im = np.array(im)
        images.append(im / 255)

        label = 1 if (row['finding'] == 'COVID-19' or row['finding'] == 'COVID-19, ARDS') else 0
        labels.append(label)
        
    return np.array(images), np.array(labels)


def TrainImageDataGenerator(shear_range=0.2, zoom_range=0.2, 
                            horizontal_flip=True):
    """A wrapper for generating ImageDataGenerator object for training data.
    
    Parameters
    ----------
    shear_range : float
        Shear Intensity (Shear angle in counter-clockwise direction in degrees).
    
    zoom_range : float
        Float or [lower, upper]. Range for random zoom.

    horizontal_flip : boolean
        Boolean. Randomly flip inputs horizontally.

    Returns
    -------
    train_generator : Keras ImageDataGenerator object
        A Keras ImageDataGenerator object with the specified arguments.
    """
    train_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip)

    return train_generator


def ValidImageDataGenerator():
    """A wrapper for generating ImageDataGenerator object for validation data."""
    return ImageDataGenerator(rescale=1./255)


def data_generator_from_dataframe(datagen, df, image_size=(256, 256), batch_size=32):
    """A function to process metadata dataframes and return generators for
    generating augmented training and validation data.
    
    Parameters
    ----------
    datagen : Keras ImageDataGenerator
        Keras ImageDataGenerator object

    df : pandas DataFrame
        Dataframe of metadata.

    image_size : tuple of int
        Size of images to scale to.

    batch_size : int
        Batch size to use when fitting.

    Returns
    -------
    generator : Keras ImageDataGenerator object
        Keras ImageDataGenerator object with a defined flow from dataframe.
    """
    # Define label 1 for COVID, 0 otherwise
    df['label'] = ((df['finding'] == 'COVID-19') | (df['finding'] == 'COVID-19, ARDS')).astype('int')
    
    # Define data generators
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory='images/',
        x_col="filename",
        y_col="label",
        target_size=image_size,
        batch_size=batch_size,
        class_mode='raw')

    return generator


if __name__ == '__main__':
    df_train, df_val = load_metadata()

    # images_train, labels_train = load_data(df_train)
    # images_val, labels_val = load_data(df_val)

    train_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True,
                                         width_shift_range=100)

    train_generator = data_generator_from_dataframe(train_generator, df_train)

    validation_generator = data_generator_from_dataframe(ValidImageDataGenerator(), df_val)

    # Generate augmented images
    for image_batch, label_batch in train_generator:
        for img, label in zip(image_batch, label_batch):
            print('Label =', label)
            plt.imshow(img)
            plt.show()
