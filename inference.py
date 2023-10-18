"""
This script is used to make predictions on live webcam feed.
It uses the trained model to predict the yoga pose being performed.
"""

import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model


def inFrame(lst):
    """
    Check if all the required landmarks are in the frame.

    Parameters:
    lst (list): List of landmarks.

    Returns:
    bool: True if all landmarks are in the frame, False otherwise.
    """
    return all(
        landmark.visibility > 0.6 for landmark in [lst[28], lst[27], lst[15], lst[16]]
    )


def load_model_and_labels():
    """
    Load the trained model and labels from disk.

    Returns:
    tuple: The trained model and the labels.
    """
    model = load_model("model.h5")
    labels = np.load("labels.npy")
    return model, labels


def predict_pose(model, labels, landmarks):
    """
    Predict the yoga pose from the landmarks using the trained model.

    Parameters:
    model (keras.Model): The trained model.
    labels (numpy.ndarray): The labels.
    landmarks (list): The landmarks.

    Returns:
    tuple: The predicted label and the confidence score.
    """
    lst = [
        (landmark.x - landmarks[0].x, landmark.y - landmarks[0].y)
        for landmark in landmarks
    ]
    prediction = model.predict(np.array(lst).reshape(1, -1))
    return labels[np.argmax(prediction)], prediction[0][np.argmax(prediction)]


def main():
    """
    Main function to run the script.
    It loads the model and labels, opens the webcam feed, and starts making predictions.
    """
    model, labels = load_model_and_labels()
    holistic = mp.solutions.pose.Pose()
    drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = cv2.blur(frame, (4, 4))

        if results.pose_landmarks and inFrame(results.pose_landmarks.landmark):
            pred, confidence = predict_pose(
                model, labels, results.pose_landmarks.landmark
            )
            if confidence > 0.75:
                cv2.putText(
                    frame, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    frame,
                    "Asana is either wrong or not trained",
                    (100, 180),
                    cv2.FONT_ITALIC,
                    1.8,
                    (0, 0, 255),
                    3,
                )
        else:
            cv2.putText(
                frame,
                "Make Sure Full body visible",
                (100, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3,
            )

        drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            connection_drawing_spec=drawing.DrawingSpec(
                color=(255, 255, 255), thickness=6
            ),
            landmark_drawing_spec=drawing.DrawingSpec(
                color=(0, 0, 255), circle_radius=3, thickness=3
            ),
        )

        cv2.imshow("window", frame)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == "__main__":
    main()
