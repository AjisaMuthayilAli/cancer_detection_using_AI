#!/usr/bin/env python
# coding: utf-8

# # Detection of cancer cells from pathological scan images

# ## Exploratory Data Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import ostrain_labels = pd.read_csv("C:\Users\clear\Documents\Jobs\labelledcancer.csv') 
# ## Cancerous images are labelled as 1 and non-cancerous images 0
# Display a subset of images with and without tumor tissue
def display_images(samples, title):
    fig, axes = plt.subplots(1, len(samples), figsize=(20, 5))
    for img_path, ax in zip(samples, axes):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.show()

# Positive images with tumor
positive_samples = train_labels[train_labels['label'] == 1]['id'].head(5).apply(lambda x: f'../input/histopathologic-cancer-detection/train/{x}.tif').tolist()
display_images(positive_samples, "Images with Tumor Tissue")

# Negative images without tumor
negative_samples = train_labels[train_labels['label'] == 0]['id'].head(5).apply(lambda x: f'../input/histopathologic-cancer-detection/train/{x}.tif').tolist()
display_images(negative_samples, "Images without Tumor Tissue")
# ## Distribution plot
plt.figure(figsize=(8, 5))
train_labels['label'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribution of Tumor (1) and No Tumor (0) Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()# Checking the resolution and contrast of a random subset of images
image_paths = train_labels['id'].sample(5).apply(lambda x: f'../input/histopathologic-cancer-detection/train/{x}.tif').tolist()

for img_path in image_paths:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Displaying image details
    print(f"Image: {img_path.split('/')[-1]}")
    print(f"Resolution: {img.shape[0]}x{img.shape[1]}")
    print(f"Contrast (max-min pixel values): {np.max(img_rgb) - np.min(img_rgb)}\n")# Adjust image sizes
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the desired dimensions
IMG_WIDTH, IMG_HEIGHT = 96, 96

def resize_image(img_path, target_width=IMG_WIDTH, target_height=IMG_HEIGHT):
    img = load_img(img_path, target_size=(target_width, target_height))
    return img_to_array(img)

# Example usage
sample_path = train_labels['id'].iloc[0]
resized_img = resize_image(f'../input/histopathologic-cancer-detection/train/{sample_path}.tif')
print(f"Resized Image Shape: {resized_img.shape}")
# #### Normalisation
def normalize_image(img):
    return img / 255.0

# Example usage
normalized_img = normalize_image(resized_img)
print(f"Pixel Range after Normalization: {np.min(normalized_img)} - {np.max(normalized_img)}")
# #### Train_test_split
from sklearn.model_selection import train_test_split

# Splitting the data into training, validation, and test sets (80%, 10%, 10%)
train_data, temp_data, train_labels, temp_labels = train_test_split(train_labels['id'].values, train_labels['label'].values, test_size=0.2, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)
# #### Image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
import matplotlib.pyplot as plt

# Data augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Display random augmented images
img = load_img(f'../input/histopathologic-cancer-detection/train/{train_data[0]}.tif')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, batch in enumerate(datagen.flow(x, batch_size=1)):
    axes[i].imshow(array_to_img(batch[0]))
    axes[i].axis('off')
    if i == 4: # Stop after displaying 5 images
        break
plt.show()
# #### Modeling
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.summary()from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# #### Model training
train_labels_df = pd.read_csv("C:\Users\clear\Documents\Jobs\labelledcancer.csv')
train_labels_df['label'] = train_labels_df['label'].astype(str)# Add the .tif extension to the 'id' column for correct file referencing
train_labels_df['id'] = train_labels_df['id'].apply(lambda x: f"{x}.tif")

# Preparing data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalize images

batch_size = 32
train_steps = 8000 // batch_size  # 8000 images for training
val_steps = 2000 // batch_size    # 2000 images for validation

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_labels_df.head(10000),
    directory='../input/histopathologic-cancer-detection/train/',
    x_col='id',
    y_col='label',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode='binary',
    batch_size=batch_size,
    subset='training'
)

val_gen = train_datagen.flow_from_dataframe(
    dataframe=train_labels_df.head(10000),
    directory='../input/histopathologic-cancer-detection/train/',
    x_col='id',
    y_col='label',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode='binary',
    batch_size=batch_size,
    subset='validation'
)
# Training the model
history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=10
)

# #### Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Adjust the val_steps
val_steps = np.ceil(len(val_gen.classes) / batch_size)

# Predict classes
val_predictions = model.predict(val_gen, steps=val_steps)
val_pred_classes = (val_predictions > 0.5).astype(int).flatten()

# True labels
true_labels = val_gen.classes

# Ensure the lengths match
val_pred_classes = val_pred_classes[:len(true_labels)]
# Calculate metrics
accuracy = accuracy_score(true_labels, val_pred_classes)
precision = precision_score(true_labels, val_pred_classes)
recall = recall_score(true_labels, val_pred_classes)
f1 = f1_score(true_labels, val_pred_classes)
roc_auc = roc_auc_score(true_labels, val_predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")# checking for overfitting
# Plotting training and validation loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.tight_layout()
plt.show()# Extracting misclassified image indices
misclassified_idx = np.where(val_pred_classes != true_labels)[0]

# Displaying a subset of misclassified images
sample_misclassified = np.random.choice(misclassified_idx, 5)

plt.figure(figsize=(20, 5))
for i, idx in enumerate(sample_misclassified):
    img = load_img(val_gen.filepaths[idx])
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"True: {true_labels[idx]}, Pred: {val_pred_classes[idx]}")
    plt.axis('off')
plt.suptitle("Misclassified Images", fontsize=16)
plt.show()
# #### Fine tuning
!pip install keras-tuner

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

def build_model(hp):
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv2D(hp.Int('input_units', min_value=32, max_value=64, step=32), (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    for i in range(hp.Int('n_layers', 1, 3)):  # adding between 1 and 3 convolutional layers
        model.add(Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=64, step=32), (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
     optimizer = Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4]))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,  # reduced number of model configurations to test
    executions_per_trial=1,
    directory='output',
    project_name='HistoPathologicCancerDetection'
)

# Train for fewer epochs during hyperparameter tuning
tuner.search(train_gen, epochs=5, validation_data=val_gen)from tensorflow.keras.applications import VGG16

# Load the VGG16 model with weights pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a custom model on top
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])#Regularisation

from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2

# Load the VGG16 model with weights pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)

# Convolutional layers
# Removed pooling layers and adjusted convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01), padding='same'))

# Fully connected layers
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# Plotting training and validation loss and accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.show()