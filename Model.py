from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np
from sklearn.model_selection import train_test_split
import os


class Model:
    def __init__(self, model_file='trained_model.h5'):
        self.model_file = model_file
        self.model = None
        self.load_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_model(self):
        # check if the model file exists
        if os.path.exists(self.model_file):
            self.model = load_model(self.model_file)
            print("Model loaded successfully!")
        else:
            print("Model not found. Need to train the model.")

    def train(self, samples, labels, learning_rate, preference):

        inputs, outputs = self.prepare_data(samples, labels)
        x_train, x_val, y_train, y_val = train_test_split(inputs, outputs, test_size=0.5, random_state=42)

        if self.model is None:
            self.model = self.build_model()
            self.model.fit(x_train, y_train, epochs=20, batch_size=32,
                           validation_data=(x_val, y_val))
            print(f"Model trained on {len(x_train)} samples.")

        self.save(self.model_file)

    def save(self, filepath):
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save.")

    def prepare_data(self, samples, labels):

        inputs = np.array(samples)
        outputs = np.array(labels)
        return inputs, outputs


    def predict(self, image):

        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image)
        predicted_label = np.argmax(predictions)
        predicted_probability = np.max(predictions)
        return predicted_label, predicted_probability
