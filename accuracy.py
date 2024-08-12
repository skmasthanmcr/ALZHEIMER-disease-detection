import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models

# Suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the saved model
model = models.load_model('capsulenet_model.keras')

# Define class names for display
def get_class_name(number):
    if number == 0:
        return 'Non Demented'
    elif number == 1:
        return 'Mild Dementia'
    elif number == 2:
        return 'Moderate Dementia'
    elif number == 3:
        return 'Very Mild Dementia'
    else:
        return 'Error in Prediction'

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 128, 128, 3)
    return img_array

# Define paths to the directories with images for each class
path1 = 'D:/vs code/datasets/Alzheimer/Non Demented'
path2 = 'D:/vs code/datasets/Alzheimer/Mild Dementia'
path3 = 'D:/vs code/datasets/Alzheimer/Moderate Dementia'
path4 = 'D:/vs code/datasets/Alzheimer/Very mild Dementia'

# Function to iterate through images and calculate accuracy
def calculate_accuracy_for_path(image_path):
    data = []
    true_labels = []
    predicted_labels = []

    # Get true label based on directory name
    if 'Non Demented' in image_path:
        true_label = 0
    elif 'Mild Dementia' in image_path:
        true_label = 1
    elif 'Moderate Dementia' in image_path:
        true_label = 2
    elif 'Very mild Dementia' in image_path:
        true_label = 3
    else:
        return 0.0  # If directory name doesn't match expected classes, return 0 accuracy

    # Iterate through images in the directory
    for filename in os.listdir(image_path):
        img_path = os.path.join(image_path, filename)
        img = load_and_preprocess_image(img_path)

        # Predict class of the image
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Append true and predicted labels
        true_labels.append(true_label)
        predicted_labels.append(predicted_class)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

# Calculate accuracy for each path
accuracy1 = calculate_accuracy_for_path(path1)
accuracy2 = calculate_accuracy_for_path(path2)
accuracy3 = calculate_accuracy_for_path(path3)
accuracy4 = calculate_accuracy_for_path(path4)

# Print accuracies
print(f"Accuracy for Non Demented images: {accuracy1:.4f}")
print(f"Accuracy for Mild Dementia images: {accuracy2:.4f}")
print(f"Accuracy for Moderate Dementia images: {accuracy3:.4f}")
print(f"Accuracy for Very Mild Dementia images: {accuracy4:.4f}")

