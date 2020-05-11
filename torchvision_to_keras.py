import torch 
import torchxrayvision as xrv
import numpy as np
from pytorch2keras import pytorch_to_keras


def torchxrayvision_to_keras(weights="chex", image_size=(224, 224), batch=32,
                             num_channel=1):
    """Converts torchxrayvision DenseNet model with a certain set of weights to 
    a Keras model and saves it.
    
    Parameters
    ----------
    weights : string {"all", "kaggle", "nih", "pc", "chex", "mimic_ch", 
                      "mimic_nb"}
        Weights to load into model. 
    
    image_size : tuple of int
        Size of input images.

    batch : int
        Batch size of placeholder tensor for model input.
    """

    # Load Densenet model with weights
    model = xrv.models.DenseNet(weights=weights)

    # Define input variable
    input_np = np.random.uniform(0, 1, (batch, 1, image_size[0], image_size[1]))
    input_var = torch.autograd.Variable(torch.FloatTensor(input_np))

    # Convert model
    k_model = pytorch_to_keras(model, input_var, [(1, image_size[0], image_size[1])], verbose=True)  

    # Save model
    k_model.save('models/' + weights + '_' + str(image_size[0]) + '_' + str(batch) + '_keras.h5')


if __name__=='__main__':

    image_size = (224, 224)
    batch = 16

    for w in ["all", "kaggle", "nih", "pc", "chex", "mimic_ch", "mimic_nb"]:
        torchxrayvision_to_keras(weights=w, image_size=image_size, batch=batch)
