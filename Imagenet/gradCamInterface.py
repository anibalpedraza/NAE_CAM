"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2021/03/07
Description: How to obtain a class activation heatmap for an image classification model.
Accelerator: GPU
https://keras.io/examples/vision/grad_cam/
"""
"""
Adaptado para el trabajo final de grado de Nerea Leon (Nerea.Leon@alu.uclm.es)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
# Display
from IPython.display import Image, display
import matplotlib.cm as cm

# Keras 3
from keras.api.models import Model
from keras.api.utils import load_img, img_to_array, array_to_img

def get_img_array_path(img_path, size):
    # `img` is a PIL image of size 299x299
    img = load_img(img_path, target_size=size)  # Actualización para Keras 3
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = img_to_array(img)  # Actualización para Keras 3
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def get_img_array(img_array):
    # `array` is a float32 Numpy array of shape (32, 32, 3)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 32, 32, 3)
    array = np.expand_dims(img_array, axis=0)
    return array
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
#https://www.youtube.com/watch?v=6YZoZ9Vtez0&ab_channel=ConnorShorten
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)) #0,1,2
    # https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap

def save_and_display_gradcam_path(img_path, heatmap, cam_path="cam.jpg", alpha=0.4, color="jet"):
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    superimposed_img = display_gradcam(img, heatmap, alpha, color)
    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4, color="jet"):
    superimposed_img = display_gradcam(img, heatmap, alpha, color)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

def display_gradcam(img, heatmap, alpha=0.4, color="jet"):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap(color)

    # Use RGB values of the colormap https://matplotlib.org/stable/tutorials/colors/colormaps.html
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)  # Actualización para Keras 3
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)  # Actualización para Keras 3

    # Superimpose the heatmap on original image
    superimposed_img_array = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img_array)  # Actualización para Keras 3

    # return Grad CAM
    return superimposed_img

def display_gray_gradcam(img, heatmap, superimposed=True):
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("gray")

    # Use RGB values of the colormap https://matplotlib.org/stable/tutorials/colors/colormaps.html
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    heatmap_img = array_to_img(jet_heatmap)
    heatmap_img = heatmap_img.resize((img.shape[1], img.shape[0]))
    heatmap_img = img_to_array(heatmap_img)

    if superimposed:
        superimposed_img_array = img*0
        # Use heatmap as filter on original image
        for channel in range(0, img.shape[2]):
            for row in range(0, img.shape[0]):
                for col in range(0, img.shape[1]):
                    superimposed_img_array[row, col, channel]= img[row, col, channel]*round(heatmap_img[row, col, channel]/255, 2)

        result = array_to_img(superimposed_img_array)
    else:
        return heatmap_img

    return result

def decode_predictions(preds_oneHotEncoder, num_classes, classes_array):
    if len(classes_array) != num_classes:
        print('Error, el numero de clases no coincide con el tamaño del array de clases')
    if preds_oneHotEncoder.shape == (10,):
        max_idx = np.where(preds_oneHotEncoder == np.amax(preds_oneHotEncoder))[0]
    else:
        max_idx = np.where(preds_oneHotEncoder == np.amax(preds_oneHotEncoder))[1]
    max_idx = max_idx[0]
    return classes_array[max_idx]
