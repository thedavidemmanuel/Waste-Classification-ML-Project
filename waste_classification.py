# -*- coding: utf-8 -*-
"""waste-classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1opecGClVO-zRR16T9AL0h5vY1P9tcQyR
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

"""# Data Augmentation and Preparation"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up training and validation directories
train_dir = '/content/dataset/dataset/DATASET/TRAIN'
test_dir = '/content/dataset/dataset/DATASET/TEST'

# Set up data generators with improved data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x/255.0,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x/255.0
)

# Initialize data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

"""# Build and Compile the Model with Regularization"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Load MobileNetV2 base model with L2 regularization
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers on top of MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  # Add L2 regularization
x = Dropout(0.3)(x)  # Adjust dropout rate
predictions = Dense(1, activation='sigmoid')(x)

# Compile the model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# View the structure of the model
model.summary()

"""# Implement Callbacks and Train the Model"""

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)

# Train the Model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs based on the training progress
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

"""# Plot Training and Validation Graphs"""

# Plot training & validation accuracy values
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

"""### Save the Trained Model"""

model_file_path = '/content/drive/MyDrive/waste_classification_model.h5'
model.save(model_file_path)
print(f"Model saved to {model_file_path}")