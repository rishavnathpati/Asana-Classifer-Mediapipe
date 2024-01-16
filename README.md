# Yoga Asana Classifier

## Description

Welcome to the Yoga Asana Classifier project. This project uses deep learning to predict the yoga pose you are performing in front of the webcam. It consists of three Python scripts: `data_collection.py`, `data_training.py`, and `inference.py`.

- **data_collection.py:** Collects training data using the Mediapipe pose detection library to detect the human body pose and saves the pose data for each frame to a .npy file.

- **data_training.py:** Trains a Dense neural network model on the collected data. The model is implemented using Keras and TensorFlow.

- **inference.py:** Makes predictions on live webcam feed using the trained model to predict the yoga pose being performed.

## Requirements

To run this project, you will need the following libraries:

- mediapipe
- keras
- tensorflow
- opencv-python
- numpy

You can install these libraries using the following command:

```bash
pip install mediapipe keras tensorflow opencv-python numpy
```

## How to Run?

### Data Collection

To collect data for a new yoga pose, run the `data_collection.py` script and provide the name of the yoga pose when prompted. The script will open a webcam feed and start collecting pose data. Make sure to perform the yoga pose in front of the webcam. The data will be saved to a .npy file in the current directory.

```bash
python data_collection.py
```

### Training

To train the model on the collected data, run the `data_training.py` script. The script will load the data from the .npy files in the current directory, train the model on the data, and save the trained model to a .h5 file.

```bash
python data_training.py
```

### Inference

To make predictions on live webcam feed, run the `inference.py` script. The script will open a webcam feed and start making predictions on the yoga pose being performed. The predicted pose will be displayed on the webcam feed.

```bash
python inference.py
```

## Configuration Parameters

The following configuration parameters are defined in the `config.py` script:

- **DATA_DIRECTORY:** The directory where the .npy files are stored.
- **MODEL_PATH:** The path to the saved model file.
- **LABELS_PATH:** The path to the saved labels file.
- **WEBCAM_INDEX:** The index of the webcam to use for data collection and inference.
- **TRAIN_TEST_SPLIT:** The fraction of data to use for training and testing.
- **EPOCHS:** The number of epochs to train the model for.

## Contributing

If you would like to contribute to this project, please feel free to fork the repository, make your changes, and open a pull request. If you have any questions, please feel free to reach out.