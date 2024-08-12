import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define CapsuleNet layers
class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))

class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.get_shape().as_list()[1])
        return tf.compat.v1.layers.batch_normalization(inputs * tf.expand_dims(mask, -1))

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return vectors / scale

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.W = self.add_weight(shape=[self.num_capsules, input_shape[1], self.dim_capsule, input_shape[2]],
                                 initializer='glorot_uniform',
                                 name='W')
        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsules, 1, 1])
        inputs_hat = tf.map_fn(lambda x: tf.keras.backend.batch_dot(x, self.W, [2, 2]), elems=inputs_tiled)
        b = tf.zeros_like(inputs_hat[:, :, :, 0])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(tf.reduce_sum(inputs_hat * tf.expand_dims(c, -1), axis=2))
            if i < self.routings - 1:
                b += tf.reduce_sum(inputs_hat * tf.expand_dims(outputs, -1), axis=-1)
        return outputs

# Define the model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load and preprocess image data
def preprocess_data():
    data = []
    result = []
    encoder = OneHotEncoder()
    encoder.fit([[0], [1], [2], [3]])

    # New dataset paths
    path1 = 'D:/vs code/datasets/Alzheimer/Non Demented'
    path2 = 'D:/vs code/datasets/Alzheimer/Mild Dementia'
    path3 = 'D:/vs code/datasets/Alzheimer/Moderate Dementia'
    path4 = 'D:/vs code/datasets/Alzheimer/Very mild Dementia'

    # Collecting file paths
    path1_files = [os.path.join(path1, filename) for filename in os.listdir(path1)[:25000]]
    path2_files = [os.path.join(path2, filename) for filename in os.listdir(path2)[:5000]]
    path3_files = [os.path.join(path3, filename) for filename in os.listdir(path3)[:488]]
    path4_files = [os.path.join(path4, filename) for filename in os.listdir(path4)[:10000]]

    # Process images and labels
    for path in path1_files:
        img = Image.open(path).resize((128, 128))
        img_array = np.array(img)
        if img_array.shape == (128, 128, 3):
            data.append(img_array)
            result.append(encoder.transform([[0]]).toarray())

    for path in path2_files:
        img = Image.open(path).resize((128, 128))
        img_array = np.array(img)
        if img_array.shape == (128, 128, 3):
            data.append(img_array)
            result.append(encoder.transform([[1]]).toarray())

    for path in path3_files:
        img = Image.open(path).resize((128, 128))
        img_array = np.array(img)
        if img_array.shape == (128, 128, 3):
            data.append(img_array)
            result.append(encoder.transform([[2]]).toarray())

    for path in path4_files:
        img = Image.open(path).resize((128, 128))
        img_array = np.array(img)
        if img_array.shape == (128, 128, 3):
            data.append(img_array)
            result.append(encoder.transform([[3]]).toarray())

    data = np.array(data)
    result = np.array(result)
    result = result.reshape((len(result), -1))  # Ensure reshaping is correct

    return data, result

# Preprocess data
data, result = preprocess_data()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.30, shuffle=True, random_state=42)

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_data=(x_test, y_test))

# Save the trained model in the native Keras format
model.save('capsulenet_model.keras')
