import mediapipe as mp
import numpy as np
import cv2
import config

# Constants
VISIBILITY_THRESHOLD = 0.6
DATA_COLLECTION_SIZE = 80  # Number of frames to collect
ESC_KEY = 27  # ASCII code for the ESC key


def inFrame(lst):
    # Check if the required landmarks are visible enough
    return all(lst[i].visibility > VISIBILITY_THRESHOLD for i in [28, 27, 15, 16])


def extract_landmarks(landmarks):
    # Extract normalized landmark coordinates relative to the first landmark
    lst = []
    for i in landmarks:
        lst.append(i.x - landmarks[0].x)
        lst.append(i.y - landmarks[0].y)
    return lst


# Initialize camera
cap = cv2.VideoCapture(config.WEBCAM_INDEX)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

name = input("Enter the name of the Asana: ")

# Initialize Mediapipe pose detection
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        lst = extract_landmarks(res.pose_landmarks.landmark)
        X.append(lst)
        data_size += 1

    else:
        cv2.putText(
            frm,
            "Make Sure Full body visible",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

    cv2.putText(
        frm, str(data_size), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.imshow("window", frm)

    if cv2.waitKey(1) == ESC_KEY or data_size >= DATA_COLLECTION_SIZE:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the collected data
np.save(f"{config.DATA_DIRECTORY}/{name}.npy", np.array(X))

print(f"Data saved for {name}. Shape of the data: {np.array(X).shape}")
