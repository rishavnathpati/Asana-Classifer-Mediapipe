import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the training data from .npy files in the current directory.
    
    Each .npy file contains data for a specific label. The name of the file (excluding the extension) is used as the label.
    The function returns the training data, the corresponding labels (one-hot encoded), a list of label names, and a dictionary mapping label names to indices.
    """
    X, y, labels = [], [], []  # Initialize lists to hold data, labels, and label names
    dictionary = {}  # Initialize dictionary to hold label name to index mapping
    for idx, file in enumerate(os.listdir()):  # Loop over all files in the current directory
        name, ext = os.path.splitext(file)  # Split the file name into name and extension
        if ext == ".npy" and name != "labels":  # If the file is a .npy file and not the labels file
            data = np.load(file)  # Load the data from the file
            X.append(data)  # Append the data to the data list
            y.append(np.full(data.shape[0], idx))  # Append the corresponding labels to the labels list
            labels.append(name)  # Append the label name to the label names list
            dictionary[name] = idx  # Add the label name to index mapping to the dictionary
    X = np.concatenate(X)  # Concatenate all data into a single numpy array
    y = to_categorical(np.concatenate(y))  # Concatenate and one-hot encode all labels into a single numpy array
    return X, y, labels, dictionary  # Return the data, labels, label names, and dictionary

def shuffle_data(X, y):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.0)
    return X_train, y_train

def create_model(input_shape, output_shape):
    ip = Input(shape=input_shape)
    m = Dense(128, activation="tanh")(ip)
    m = Dense(64, activation="tanh")(m)
    op = Dense(output_shape, activation="softmax")(m)
    model = Model(inputs=ip, outputs=op)
    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
    return model

def train_model(model, X, y):
    model.fit(X, y, epochs=80)
    model.save("model.h5")

def save_labels(labels):
    np.save("labels.npy", np.array(labels))

def main():
    X, y, labels, dictionary = load_data()
    X, y = shuffle_data(X, y)
    model = create_model(X.shape[1], y.shape[1])
    train_model(model, X, y)
    save_labels(labels)

if __name__ == "__main__":
    main()
