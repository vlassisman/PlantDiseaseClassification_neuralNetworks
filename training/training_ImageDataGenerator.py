import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 256
CHANNELS = 3

# TRAINING
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10
)

train_generator = train_datagen.flow_from_directory(
    'C:/dev/git/PlantDiseaseClassification_neuralNetworks/training/dataset/train',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='sparse'
)

# VALIDATION
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True
)

validation_generator = validation_datagen.flow_from_directory(
    'C:/dev/git/PlantDiseaseClassification_neuralNetworks/training/dataset/val',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode="sparse"
)

# TEST
test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True
)

test_generator = test_datagen.flow_from_directory(
    'C:/dev/git/PlantDiseaseClassification_neuralNetworks/training/dataset/test',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode="sparse"
)

# MODEL
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),

    # convolution - pooling stack 
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # flatten data - dense layer - output
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')  ## normalization
])

model.summary()

model.compile(
    optimizer='adam',  ## sthg like gradient descent
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# NETWORK TRAINING
history = model.fit(
    train_generator,
    steps_per_epoch=47,
    batch_size=32,
    verbose=1, # prints whats going on
    validation_data=validation_generator,
    validation_steps=6,
    epochs=20
)

model.save('../_potatoes.h5')
