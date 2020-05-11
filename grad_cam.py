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
from vis.visualization import visualize_cam, visualize_saliency, overlay
from keras import activations
from vis.utils import utils


def plot_gradcam_saliency(im_files, model_types=None, modifier=None):
    """Plot gradcam and saliency heatmaps for a given image and model type.
    
    Parameters
    ----------
    im_files : list of str
        Path to image file to open.

    model_types: list of str {'dense', 'dense_all', 'vgg16'}
        Type of model to generate maps for. If `None` then will use benchmark
        classifier with no data pre-processing.

    modifier : str {'deconv', 'rectified', 'relu', 'guided'}
        Backprop modifier to use.
    """
    row_nums = len(im_files)
    col_nums = 2 * len(model_types) + 1

    fig_len = row_nums * col_nums

    fig, axes = plt.subplots(row_nums, col_nums, figsize=(28, 4))

    for im_idx, im_file in enumerate(im_files):

        img = image.load_img(im_file, target_size=(224, 224))
        img1 = image.img_to_array(img)
        img1 = np.expand_dims(img1, axis=0)
        
        img_init=utils.load_img(im_file, target_size=(224, 224))

        for ax in axes[im_idx]:
            ax.imshow(img)
        for ax in axes[im_idx, 1:]:
            ax.axis('off')
        
        axes[im_idx, 0].set_ylabel('COVID = ' + str(1 - im_idx))
           
        for mod_idx, model_type in enumerate(model_types):

            if model_type == 'dense':
                model = load_model('models/chexnet_2020_05_07_17_34_41.h5')
            elif model_type == 'dense_all':
                model = load_model('models/chexnet_all_layers_2020_05_08_14_59_30.h5')
            elif model_type == 'vgg16':
                model = load_model('models/vgg16_2020_05_09_15_49_56.h5')
            elif model_type == 'dense_imagenet':
                model = load_model('models/dense_imagenet_2020_05_08_18_15_13.h5')
            else:
                model = load_model('models/model_aug_bench_2020_05_07_17_56_23.h5')
        
            # Prediction before replacing layer
            pred = model.predict(np.expand_dims(img_init, axis=0)/255)[0][0]

            # Replace final layer with linear
            model.layers[-1].activation = activations.linear
            model = utils.apply_modifications(model)

            if model_type == 'dense' or model_type == 'dense_all' or model_type=='dense_imagenet':
                img1 = dense_preprocess(img1)
            elif model_type == 'vgg16':
                img1 = vgg_preprocess(img1)
                
            if model_type == 'dense' or model_type == 'dense_all':
                p_idx = 423  # Last convolutional layer
            else:
                p_idx = None

            grads = visualize_saliency(model, -1, filter_indices=0, seed_input=img1[0,:,:,:],
                                    backprop_modifier=modifier)

            heatmap = visualize_cam(model, layer_idx=-1, filter_indices=0, seed_input=img1[0,:,:,:],
                                            backprop_modifier=modifier, penultimate_layer_idx=p_idx)

            im = axes[im_idx, 2*mod_idx + 1].imshow(heatmap, cmap='jet', alpha=0.5)
            im = axes[im_idx, 2*mod_idx + 2].imshow(grads, cmap='jet', alpha=0.5)

            if model_type == 'dense':
                title = 'CheXNet'
            elif model_type == 'dense_all':
                title = 'CheXNet (All Layers)'
            elif model_type == 'dense_imagenet':
                title = 'DenseNet'
            elif model_type == 'vgg16':
                title = 'VGG16'
            else:
                title = 'Benchmark CNN'
            
            axes[im_idx, 2*mod_idx + 1].set_title('{} : P(COVID) = {:.3f}'.format(title, pred))
    
    plt.tight_layout()
    cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)
    plt.show()


if __name__ == '__main__':
    df_train, df_val = load_metadata()

    # images_train, labels_train = load_data(df_train)
    images_val, labels_val = load_data(df_val)

    model_types = ['bench', 'vgg16', 'dense']

    im_files = ["images/1-s2.0-S0929664620300449-gr2_lrg-a.jpg",
                "images/F051E018-DAD1-4506-AD43-BE4CA29E960B.jpeg"]
   
    plot_gradcam_saliency(im_files, model_types)
 