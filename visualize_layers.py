from load_data import load_data, load_metadata, ValidImageDataGenerator, data_generator_from_dataframe
import tensorflow as tf
import vis ## keras-vis
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from keras.applications.densenet import preprocess_input as dense_preprocess
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Input
from keras.models import Model
from keras.preprocessing import image
from keras.models import load_model
from keras import activations
from vis.visualization import visualize_activation, overlay, get_num_filters
from vis.utils import utils
from vis.input_modifiers import Jitter
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator
from PIL import Image

# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def visualize_layers(model_types, max_iter=500):

    col_nums = len(model_types)
    if col_nums == 1:
        col_nums = 2
    fig, axes = plt.subplots(1, col_nums, figsize=(21, 6))

    for i, model_type in enumerate(model_types):
        if model_type == 'dense':
            model = load_model('models/chexnet_2020_05_07_17_34_41.h5')
            title = 'CheXNet'
        elif model_type == 'dense_all':
            model = load_model('models/chexnet_all_layers_2020_05_08_14_59_30.h5')
            title = 'CheXNet (All Layers)'
        elif model_type == 'vgg16':
            model = load_model('models/vgg16_2020_05_09_15_49_56.h5')
            title = 'VGG16'
        elif model_type == 'dense_imagenet':
            model = load_model('models/dense_imagenet_2020_05_08_18_15_13.h5')
            title = 'DenseNet'
        else:
            model = load_model('models/model_aug_bench_2020_05_07_17_56_23.h5')
            title = 'Benchmark CNN'
        
        # Replace final layer with linear activation
        model.layers[-1].activation = activations.linear
        model = utils.apply_modifications(model)

        img = visualize_activation(model, -1, filter_indices=0, tv_weight=0.,
                                   max_iter=max_iter, input_modifiers=[Jitter(0.1)])
        axes[i].imshow(img)
        axes[i].set_title('{}'.format(title))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def callback_gif(model_type, layer_name='conv2d_1', max_iter=100):
        
    if model_type == 'dense':
        model = load_model('models/chexnet_2020_05_07_17_34_41.h5')
        title = 'CheXNet'
    elif model_type == 'dense_all':
        model = load_model('models/chexnet_all_layers_2020_05_08_14_59_30.h5')
        title = 'CheXNet (All Layers)'
    elif model_type == 'vgg16':
        model = load_model('models/vgg16_2020_05_09_15_49_56.h5')
        title = 'VGG16'
    elif model_type == 'dense_imagenet':
        model = load_model('models/dense_imagenet_2020_05_08_18_15_13.h5')
        title = 'DenseNet'
    else:
        model = load_model('models/model_aug_bench_2020_05_07_17_56_23.h5')
        title = 'Benchmark CNN'
    
    # Replace final layer with linear activation
    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)
    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)

    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    output_class = 0

    # model_input = utils.random_array((1, 224, 224, 3), mean=127.5, std=0.05*255)

    losses = [
        (ActivationMaximization(layer_dict[layer_name], output_class), 2),
        (LPNorm(model.input), 10),
        (TotalVariation(model.input), 10)
    ]
    opt = Optimizer(model.input, losses)
    
    im = opt.minimize(max_iter=max_iter, verbose=True, input_modifiers=[Jitter()], callbacks=[GifGenerator('opt_progress')])
    
    plt.imshow(im)
    plt.show()


def visualize_gif(model_type, max_iter=500, jitter=None):

    if model_type == 'dense':
        model = load_model('models/chexnet_2020_05_07_17_34_41.h5')
        title = 'CheXNet'
    elif model_type == 'dense_all':
        model = load_model('models/chexnet_all_layers_2020_05_08_14_59_30.h5')
        title = 'CheXNet (All Layers)'
    elif model_type == 'vgg16':
        model = load_model('models/vgg16_2020_05_09_15_49_56.h5')
        title = 'VGG16'
    elif model_type == 'dense_imagenet':
        model = load_model('models/dense_imagenet_2020_05_08_18_15_13.h5')
        title = 'DenseNet'
    else:
        model = load_model('models/model_aug_bench_2020_05_07_17_56_23.h5')
        title = 'Benchmark CNN'
    
    # Replace final layer with linear activation
    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)

    np.random.seed(0)

    input_noise = utils.random_array((224, 224, 3), mean=127.5, std=0.05*255)

    for i in range(1, max_iter):
        if jitter is None:
            img = visualize_activation(model, -1, filter_indices=0, max_iter=i, 
                                        seed_input=input_noise)
        else:
            img = visualize_activation(model, -1, filter_indices=0, max_iter=i, 
                                   input_modifiers=[Jitter()], seed_input=input_noise, 
                                   tv_weight=.0)

        im = Image.fromarray(img)
        
        if jitter is None:
            im.save('figures/gif/{}/{}.png'.format(model_type, i))
        else:    
            im.save('figures/gif/jitter_{}/{}.png'.format(model_type, i))


def visualize_filters(model_type, layer_name):
    if model_type == 'dense':
        model = load_model('models/chexnet_2020_05_07_17_34_41.h5')
        title = 'CheXNet'
    elif model_type == 'dense_all':
        model = load_model('models/chexnet_all_layers_2020_05_08_14_59_30.h5')
        title = 'CheXNet (All Layers)'
    elif model_type == 'vgg16':
        model = load_model('models/vgg16_2020_05_09_15_49_56.h5')
        title = 'VGG16'
    elif model_type == 'dense_imagenet':
        model = load_model('models/dense_imagenet_2020_05_08_18_15_13.h5')
        title = 'DenseNet'
    else:
        model = load_model('models/model_aug_bench_2020_05_07_17_56_23.h5')
        title = 'Benchmark CNN'
    
    print(model.summary())

    # Replace final layer with linear activation
    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)

    layer_idx = utils.find_layer_idx(model, layer_name)

    # Visualize all filters in this layer.
    filters = np.arange(get_num_filters(model.layers[layer_idx]))

    # Generate input image for each filter.
    vis_images = []
    for idx in filters:
        img = visualize_activation(model, layer_idx, filter_indices=idx)
        
        # Utility to overlay text on image.
        img = utils.draw_text(img, 'Filter {}'.format(idx))    
        vis_images.append(img)

    # Generate stitched image palette with 8 cols.
    stitched = utils.stitch_images(vis_images, cols=8)    
    plt.axis('off')
    plt.imshow(stitched)
    plt.title(layer_name)
    plt.show()


if __name__ == '__main__':

    # model_types = ['bench', 'vgg16', 'dense']

    # visualize_layers(model_types=model_types, max_iter=4096)
 
    # model_type = 'vgg16'
    # layer_name = 'block5_conv3', 'conv2_1'  # VGG16
    # layer_name = 'block5_conv3'  #, 'conv2_1'  # VGG16
    # visualize_filters(model_type=model_type, layer_name=layer_name)

    model_type = 'dense'
    visualize_gif(model_type=model_type, max_iter=100, jitter='aye')

    # callback_gif(model_type='dense')