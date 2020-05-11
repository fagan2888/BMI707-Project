# transfer_learning.py
# A script containing functions for training transfer learning models on COVID
# chest X-ray data.
# Authors: Matthew West <mwest@hsph.harvard.edu>


import datetime
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from load_data import load_data, load_metadata, data_generator_from_dataframe
from sklearn.metrics import roc_auc_score


def train_VGG(train, test, epochs=12, save=False):
    """Train a VGG16 model using ImageNet weights."""
    
    steps_per_epoch = 2000//16
    validation_steps = 600//16

    model_vgg = VGG16(weights='imagenet', include_top=False, 
                      input_shape=(224, 224, 3))

    for layer in model_vgg.layers:
        layer.trainable = False

    x = Conv2D(64, (3, 3))(model_vgg.output)
    x = GlobalAveragePooling2D()(x)
    # x = Dense(128, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model_vgg = Model(inputs=model_vgg.input, outputs=output)

    model_vgg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model_vgg.fit(train, validation_data=test, epochs=epochs, 
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'models/vgg16_' + now + '.h5'
        model_vgg.save(filepath)

    return model_vgg


def train_dense_imagenet(train, test, epochs=12, save=False):
    """Train a VGG16 model using ImageNet weights."""
    
    steps_per_epoch = 2000//16
    validation_steps = 600//16

    model_dense_imagenet = DenseNet121(weights='imagenet', include_top=False,
                                       input_shape=(224, 224, 3))

    for layer in model_dense_imagenet.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(model_dense_imagenet.output)
    output = Dense(1, activation='sigmoid')(x)

    model_dense_imagenet = Model(inputs=model_dense_imagenet.input, outputs=output)

    model_dense_imagenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model_dense_imagenet.fit(train, validation_data=test, epochs=epochs, 
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'models/dense_imagenet_' + now + '.h5'
        model_dense_imagenet.save(filepath)

    return model_dense_imagenet


def train_VGG_all_layers(train, test, epochs=12, save=False):
    """Train a VGG16 model using ImageNet weights, without freezing layers."""
    
    steps_per_epoch = 2000//16
    validation_steps = 600//16

    model_vgg = VGG16(weights='imagenet', include_top=False)

    x = GlobalAveragePooling2D()(model_vgg.output)
    output = Dense(1, activation='sigmoid')(x)

    model_vgg = Model(inputs=model_vgg.input, outputs=output)

    model_vgg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model_vgg.fit(train, validation_data=test, epochs=epochs, 
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'models/vgg16_all_layers_' + now + '.h5'
        model_vgg.save(filepath)

    return model_vgg


def train_resnet(train, test, epochs=12, save=False):
    """Train a resnet model using ImageNet weights."""
    
    steps_per_epoch = 200//16
    validation_steps = 60//16

    model_resnet = ResNet50(weights='imagenet', include_top=False)
    
    for layer in model_resnet.layers:
        layer.trainable = False

    x = Conv2D(64, (3, 3))(model_resnet.output)
    x = GlobalAveragePooling2D()(x)
    # x = Dense(128, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model_resnet = Model(inputs=model_resnet.input, outputs=output)

    model_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model_resnet.summary())
    model_resnet.fit(train, validation_data=test, epochs=epochs, 
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'models/resnet_' + now + '.h5'
        model_resnet.save(filepath)

    return model_resnet


def train_CheXNet(train, test, epochs=12, save=False):
    """Train a DenseNet model using CheXNet weights, courtesy of 
    <https://github.com/brucechou1983/CheXNet-Keras>.
    """
    
    steps_per_epoch = 200//1
    validation_steps = 60//1

    model_chexnet = DenseNet121(weights='models/brucechou1983_CheXNet_Keras_0.3.0_weights.h5', 
                             include_top=True, classes=14)
    
    for layer in model_chexnet.layers:
        layer.trainable = False

    x = Conv2D(64, (3, 3))(model_chexnet.layers[-3].output)
    x = GlobalAveragePooling2D()(x)

    output = Dense(1, activation='sigmoid')(x)

    model_chexnet = Model(inputs=model_chexnet.input, outputs=output)

    model_chexnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model_chexnet.summary())
    model_chexnet.fit(train, validation_data=test, epochs=epochs, 
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'models/chexnet_' + now + '.h5'
        model_chexnet.save(filepath)

    return model_chexnet


def train_CheXNet_all_layers(train, test, epochs=12, save=False):
    """Train a DenseNet model using CheXNet weights, courtesy of 
    <https://github.com/brucechou1983/CheXNet-Keras>, but don't freeze any of
    them.
    """
    
    steps_per_epoch = 200
    validation_steps = 60

    model_chexnet = DenseNet121(weights='models/brucechou1983_CheXNet_Keras_0.3.0_weights.h5', 
                             include_top=True, classes=14)
    
    x = GlobalAveragePooling2D()(model_chexnet.layers[-3].output)

    output = Dense(1, activation='sigmoid')(x)

    model_chexnet = Model(inputs=model_chexnet.input, outputs=output)

    model_chexnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model_chexnet.summary())
    model_chexnet.fit(train, validation_data=test, epochs=epochs, 
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'models/chexnet_all_layers_' + now + '.h5'
        model_chexnet.save(filepath)

    return model_chexnet


if __name__ == '__main__':
    df_train, df_val = load_metadata()
    train_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True,
                                         width_shift_range=1,
                                         brightness_range=[0.5, 1.2])

    train_generator = data_generator_from_dataframe(train_generator, df_train,
                                                    image_size=(224,224),
                                                    batch_size=16)
    val_generator = data_generator_from_dataframe(ImageDataGenerator(rescale=1./255),
                                                    df_val, image_size=(224,224),
                                                    batch_size=16)


    model = train_VGG(train_generator, val_generator, save=True, epochs=12)
    # model = train_resnet(train_generator, val_generator, save=True, epochs=12)
    # model = train_CheXNet(train_generator, val_generator, save=True, epochs=24)
    # model = train_CheXNet_all_layers(train_generator, val_generator, save=True, epochs=24)
    # model = train_VGG_all_layers(train_generator, val_generator, save=True, epochs=24)
    # model = train_dense_imagenet(train_generator, val_generator, save=True, epochs=12)

    # model = load_model('models/chexnet_2020_05_07_15_32_11.h5')
    images_val, labels_val = load_data(df_val)

    y_probs = model.predict(images_val)
    y_pred = [1 if p > 0.5 else 0 for p in y_probs]
    print(labels_val)
    print(y_pred)
    # print(y_probs.T)
    print(roc_auc_score(labels_val, y_probs))