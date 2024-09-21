from argparse import ArgumentParser

import numpy as np
import cv2
# import os
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.models import hub

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope



parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output high res images.')


def main():
    args = parser.parse_args()

    # Get all image paths
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]

    # Change model input shape to accept all size inputs
    # model = keras.models.load_model((r'C:\Users\aabso\OneDrive\Desktop\Capstone Project\Fast-SRGAN-master\Fast-SRGAN-master\models\generator.h5'),custom_objects={'KerasLayer':'hub.KerasLayer'})
    # Change model input shape to accept all size inputs(None, 1078, 1918, 3)

    with custom_object_scope({'CustomLayer': 10}):
        model = load_model(
            r'C:\Users\aabso\OneDrive\Desktop\Capstone Project\Fast-SRGAN-master\Fast-SRGAN-master\models\generator.h5')

    inputs = keras.Input((None, None, 3))

    resized_images = tf.image.resize(inputs, [96, 96])
    output = model(resized_images)
    # output = model(inputs)
    
    model = keras.models.Model(inputs, output)

    # Loop over all images
    for image_path in image_paths:
        
        # Read image
        low_res = cv2.imread(image_path, 1)

        # Convert to RGB (opencv uses BGR as default)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

        # Rescale to 0-1.
        low_res = low_res / 255.0

        # Get super resolution image
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]

        # Rescale values in range 0-255
        sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

        # Convert back to BGR for opencv
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        # Save the results:
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image_path)), sr)


if __name__ == '__main__':
    main()
