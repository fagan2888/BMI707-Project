# benchmark_classifier.py
# A script containing functions for training benchmark classifiers from scratch.
# Authors: Matthew West <mwest@hsph.harvard.edu>, Hillary Tsang, Anjali Jha


import datetime
from load_data import load_data, load_metadata, data_generator_from_dataframe, ValidImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


def create_model():
    """Create Keras model using Sequential API.
    
    Returns
    -------
    model : Keras Sequential object
        Keras Sequential model following the specified architecture.
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def train_benchmark(model, X_train, X_val, y_train, y_val, epochs=12, 
                    batch_size=12, save=False):
    """A function to train a benchmark CNN in Keras.
    
    Parameters
    ----------
    model : Keras Sequential model
        Model created using the `create_model` function.
    
    X_train : array-like
        NumPy array of RGB training images.

    X_val : array-like
        NumPy array of RGB validation images.

    y_train : array-like
        List of training labels, 1 for COVID, 0 for non-COVID.

    y_val : array-like
        List of validation labels, 1 for COVID, 0 for non-COVID.

    epochs : int
        Number of training epochs.
    
    batch_size : int
        Batch size for training.

    save : boolean
        Whether or not to save model.
    
    Returns
    -------
    model : Keras Sequential Model
        A trained Keras model. 
    """    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, 
              batch_size=batch_size, verbose=1)

    # Save model with timestamp as name
    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'model_bench_' + now + '.h5'
        model.save(filepath)

    return model


def train_augmented_benchmark(model, train_generator, validation_generator,
                              epochs=12, steps_per_epoch=6, validation_steps=10,
                              save=False):
    """A function to train a benchmark CNN in Keras with augmented data.
    
    Parameters
    ----------
    model : Keras Sequential model
        Model created using the `create_model` function.

    train_generator : Keras ImageDataGenerator object
        Object to generate batches of augmented tensor image data for training.

    validation_generator : Keras ImageDataGenerator object
        Object to generate batches of augmented tensor image data for validation.

    epochs : int
        Number of training epochs.
    
    steps_per_epoch : int
        Number of steps per epoch.

    validation_steps : int
        Number of validation steps.

    save : boolean
        Whether or not to save model.
    
    Returns
    -------
    model : Keras Sequential Model
        A trained Keras model. 
    """    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

    # Train model
    model.fit_generator(train_generator, validation_data=validation_generator, 
                        epochs=epochs, steps_per_epoch=steps_per_epoch, 
                        validation_steps=validation_steps)

    # Save model with timestamp as name
    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'model_aug_bench_' + now + '.h5'
        model.save(filepath)

    return model


if __name__ == '__main__':
    df_train, df_val = load_metadata()
    # images_train, labels_train = load_data(df_train)
    # images_val, labels_val = load_data(df_val)

    # model = create_model()
    # train_benchmark(model, images_train, images_val, labels_train, labels_val, 
    #                 epochs=12, batch_size=32, save=False)

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,
                                         width_shift_range=1)

    train_generator = data_generator_from_dataframe(train_datagen, df_train)

    validation_generator = data_generator_from_dataframe(ValidImageDataGenerator(), df_val)

    model = create_model()
    train_augmented_benchmark(model, train_generator, validation_generator,
                              epochs=12, steps_per_epoch=12,
                              save=False)

    # Load saved model
    # model = load_model('model_bench_2020_04_22_22_58_23.h5')    
    # print(model.summary())
    # print(model.predict_classes(images_val))