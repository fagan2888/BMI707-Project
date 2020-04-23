# benchmark_classifier.py
# A script containing functions for training benchmark classifiers from scratch.
# Authors: Matthew West <mwest@hsph.harvard.edu>, Hillary Tsang, Anjali Jha


import datetime
from load_data import load_data
from load_data import load_metadata
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D


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
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
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
    """    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, 
              batch_size=batch_size, verbose=1)

    # # Evaluation metrics
    # scores = model.evaluate(X_val, y_val, verbose=0)
    # print(scores)
    # print(y_val)
    # print(model.predict_classes(X_val, verbose=1))

    # Save model with timestamp as name
    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'model_bench_' + now + '.h5'
        model.save(filepath)


if __name__ == '__main__':
    df_train, df_val = load_metadata()
    images_train, labels_train = load_data(df_train)
    images_val, labels_val = load_data(df_val)

    model = create_model()
    train_benchmark(model, images_train, images_val, labels_train, labels_val, 
                    epochs=1, batch_size=32, save=False)

    # Load saved model
    # model = load_model('model_bench_2020_04_22_22_58_23.h5')    
    # print(model.summary())
    # print(model.predict_classes(images_val))