from IPython import display
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time

'''
Function: load_mnist_dataset
Args: N/A
Return: the MNIST dataset
Description:
'''
def load_mnist_dataset():
    return tf.keras.datasets.mnist.load_data()

'''
Function: preprocess_images
Args: image_set
Return:
Description:
'''
def preprocess_images(image_set):
    prepped_image_set = image_set.reshape((image_set.shape[0], 28, 28, 1)) / 255.
    return np.where(prepped_image_set > 0.5, 1.0, 0.0).astype('float32')

'''
Function: batch_and_shuffle_data
Args: image_set, set_size, batch_size
Return: batched and shuffled dataset (from image dataset)
Description:
'''
def batch_and_shuffle_data(image_set, set_size, batch_size):
    return (tf.data.Dataset.from_tensor_slices(image_set).shuffle(set_size).batch(batch_size))

'''
Fuction: MAIN
Args: N/A
Return:
Description: Utilizing MNIST datasets to feed into tensorflow based vcariational autoencoder
'''
def main():
    #loading the mnist dataset
    (train_images, _), (test_images, _) = load_mnist_dataset()
	
    #prepping the image data sets, both training and testing
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_size, batch_size, test_size = 60000, 32, 10000
    
    #shuffling the datasets to ensure randomness
    train_dataset = batch_and_shuffle_data(train_images, train_size, batch_size)
    test_dataset = batch_and_shuffle_data(test_images, test_size, batch_size)

    

if __name__ == "__main__":
    main()
