import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 256
CHANNELS = 3

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='sparse',
    save_to_dir="AugmentedImages"
)

