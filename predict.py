import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import models

# Suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define class names for display
def get_class_name(number):
    class_names = {
        0: 'Non Demented',
        1: 'Mild Dementia',
        2: 'Moderate Dementia',
        3: 'Very Mild Dementia'
    }
    return class_names.get(number, 'Error in Prediction')

# Load the saved model
model = models.load_model('capsulenet_model.keras')

# Initialize tkinter window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 128, 128, 3)
    return img_array

# GUI loop for image input and prediction
while True:
    # Prompt user for file selection using a file dialog
    input_image_path = filedialog.askopenfilename(title="Select an Image File",
                                                  filetypes=[("Image Files", "*.png *.jpg *.jpeg")])

    if not input_image_path:
        print("No file selected or operation cancelled.")
        break

    # Load and preprocess the image
    input_image = load_and_preprocess_image(input_image_path)

    # Predict the class of the input image
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    # Get the class name
    predicted_class_name = get_class_name(predicted_class)

    # Create a new tkinter window for displaying image and prediction
    gui_root = tk.Toplevel(root)
    gui_root.title("Image Classification Prediction")

    # Load the image to display in tkinter
    img_pil = Image.open(input_image_path)
    img_tk = ImageTk.PhotoImage(img_pil.resize((128, 128)))

    # Display the image
    label_img = tk.Label(gui_root, image=img_tk)
    label_img.image = img_tk  # Keep a reference to the image object
    label_img.pack(padx=20, pady=20)

    # Display predicted class and confidence
    label_pred = tk.Label(gui_root, text=f'Predicted Class: {predicted_class_name}\nConfidence: {confidence*100:.2f}%')
    label_pred.pack(padx=10, pady=10)

    # Close button
    close_button = tk.Button(gui_root, text="Close", command=gui_root.destroy)
    close_button.pack(pady=10)

    # Run the GUI main loop for this window
    gui_root.mainloop()

# Close the main tkinter window
root.destroy()
