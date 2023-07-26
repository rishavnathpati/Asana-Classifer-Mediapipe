# Yoga Asana Classifier

![Yoga Asana Classifier](yoga.png)

## Description
Welcome to my new deep learning project "Yoga Asana Classifier" or "Pose Classifier." This project can predict the yoga pose you are performing in front of the webcam. It consists of three Python scripts: Data Collection, Data Training, and Inference.

I utilized the Mediapipe pose detection library to detect the human body pose. Then, I created a simple Dense neural network model using Keras and trained it on the collected data. Finally, I used the inference script to make predictions.

## Requirements
- mediapipe
- keras
- tensorflow
- opencv-python
- numpy

## How to Run?

### Data Collection
To add data, run the following command and provide the name of the yoga asana you want to add:
```bash
python data_collection.py
```

### Training
Train the model on the newly added data by running:
```bash
python data_training.py
```

### Inference
Run the inference script for live predictions:
```bash
python inference.py
```
