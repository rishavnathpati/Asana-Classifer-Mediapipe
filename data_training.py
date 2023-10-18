import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

def load_data():
    X, y, labels = [], [], []
    dictionary = {}
    for idx, file in enumerate(os.listdir()):
        name, ext = os.path.splitext(file)
        if ext == ".npy" and name != "labels":
            data = np.load(file)
            X.append(data)
            y.append(np.full(data.shape[0], idx))
            labels.append(name)
            dictionary[name] = idx
    X = np.concatenate(X)
    y = to_categorical(np.concatenate(y))
    return X, y, labels, dictionary

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
