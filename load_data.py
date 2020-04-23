# load_data.py
# A script containing functions for loading and augmenting data.
# Authors: Matthew West <mwest@hsph.harvard.edu>, Hillary Tsang, Anjali Jha

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def data_augment():
    pass


if __name__ == '__main__':
    df_train, df_val = load_metadata()
    images_train, labels_train = load_data(df_train)
    images_val, labels_val = load_data(df_val)