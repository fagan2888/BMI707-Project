import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from load_data import load_data, load_metadata, data_generator_from_dataframe
from sklearn.metrics import roc_auc_score


def train_torch_chex(train_generator, val_generator, save=False, epochs=12):
    
    steps_per_epoch = 200//16
    validation_steps = 60//16

    dense_model = load_model('models/pc_224_16_keras.h5')

    # Freeze layers in model
    for layer in dense_model.layers:
        layer.trainable = False
    
    output = Dense(1, activation='sigmoid')(dense_model.layers[-2].output)

    model = Model(inputs=dense_model.input, outputs=output)

    # Unfreeze last layers in model
    for layer in model.layers[485:]:
        layer.trainable = True
    
    print(model.summary())
    # for i, layer in enumerate(dense_model.layers):
    #     print(i, layer.name)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps)

    if save:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = 'models/chex_torch_' + now + '.h5'
        model.save(filepath)
    
    return model


if __name__ == '__main__':
    df_train, df_val = load_metadata()
    train_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True,
                                         width_shift_range=1,
                                         brightness_range=[0.5, 1.2])

    train_generator = data_generator_from_dataframe(train_generator, df_train,
                                                    image_size=(224,224),
                                                    batch_size=16,
                                                    color_mode='grayscale')
    
    val_generator = ImageDataGenerator(rescale=1./255)
    
    val_generator = data_generator_from_dataframe(val_generator, df_val, 
                                                  image_size=(224,224),
                                                  batch_size=16,
                                                  color_mode='grayscale')

    model = train_torch_chex(train_generator, val_generator, save=True, epochs=12)
    
    # model = load_model('models/chexnet_2020_05_07_15_32_11.h5')
    images_val, labels_val = load_data(df_val)

    y_probs = model.predict(images_val)
    y_pred = [1 if p > 0.5 else 0 for p in y_probs]
    print(labels_val)
    print(y_pred)
    print(roc_auc_score(labels_val, y_probs))