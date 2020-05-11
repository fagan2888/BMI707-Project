# Change keras.json to have ``"image_data_format": "channels_first"``

import os

content = {
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "image_data_format": "channels_first", 
    "backend": "tensorflow"
}

path = os.path.expanduser('~')

with open(path + '/.keras/keras.json', "w") as f:
    f.write(str(content))