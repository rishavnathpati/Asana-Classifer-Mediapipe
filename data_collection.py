import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import time

# Initialize Mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create directories to store the collected data
data_dir = 'yoga_poses_data'
os.makedirs(data_dir, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the number of frames to collect per pose
num_frames_per_pose = 200

# Flag to control the recording process
recording = False

# Function to handle the "Record" button click event
def start_recording():
    global recording
    recording = True

# Function to handle the "Exit" button click event
def exit_program():
    global recording
    recording = False
    cap.release()
    window.quit()

# Create the main window
window = tk.Tk()
window.title("Yoga Pose Recorder")
window.geometry("800x600")

# Create and pack the GUI elements
label = tk.Label(window, text="Enter the yoga pose name:")
label.pack()

pose_entry = tk.Entry(window)
pose_entry.pack()

record_button = tk.Button(window, text="Record", command=start_recording)
record_button.pack()

exit_button = tk.Button(window, text="Exit", command=exit_program)
exit_button.pack()

# Create a label to display the webcam feed
webcam_label = tk.Label(window)
webcam_label.pack()

# Data collection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the pose landmarks
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw the pose landmarks and connections on the frame
        mp_drawing.draw_landmarks(frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if all the required landmarks are visible
        required_landmarks = [mp_pose.PoseLandmark.NOSE,
                              mp_pose.PoseLandmark.LEFT_SHOULDER,
                              mp_pose.PoseLandmark.RIGHT_SHOULDER,
                              mp_pose.PoseLandmark.LEFT_ELBOW,
                              mp_pose.PoseLandmark.RIGHT_ELBOW,
                              mp_pose.PoseLandmark.LEFT_WRIST,
                              mp_pose.PoseLandmark.RIGHT_WRIST,
                              mp_pose.PoseLandmark.LEFT_HIP,
                              mp_pose.PoseLandmark.RIGHT_HIP,
                              mp_pose.PoseLandmark.LEFT_KNEE,
                              mp_pose.PoseLandmark.RIGHT_KNEE,
                              mp_pose.PoseLandmark.LEFT_ANKLE,
                              mp_pose.PoseLandmark.RIGHT_ANKLE]

        all_landmarks_visible = all(landmark.visibility > 0.5 for landmark in results.pose_landmarks.landmark if landmark.type in required_landmarks)
    else:
        all_landmarks_visible = False

    # Convert the frame to PIL Image
    img = Image.fromarray(frame_rgb)

    # Convert the PIL Image to ImageTk format
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the webcam label with the new frame
    webcam_label.imgtk = imgtk
    webcam_label.configure(image=imgtk)

    if recording:
        pose_name = pose_entry.get()
        if pose_name:
            # Create a directory for the current pose if it doesn't exist
            pose_dir = os.path.join(data_dir, pose_name)
            os.makedirs(pose_dir, exist_ok=True)

            # Delay for 5 seconds before starting the recording
            countdown = 5
            while countdown > 0:
                # Display the countdown on the frame
                cv2.putText(frame_rgb, str(countdown), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert the frame to PIL Image
                img = Image.fromarray(frame_rgb)

                # Convert the PIL Image to ImageTk format
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the webcam label with the new frame
                webcam_label.imgtk = imgtk
                webcam_label.configure(image=imgtk)

                # Update the GUI
                window.update()

                # Wait for 1 second
                time.sleep(1)

                countdown -= 1

            pose_data = []
            frame_count = 0
            while frame_count < num_frames_per_pose:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror the frame horizontally
                frame = cv2.flip(frame, 1)

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect the pose landmarks
                results = pose.process(frame_rgb)

                if results.pose_landmarks and all_landmarks_visible:
                    # Draw the pose landmarks and connections on the frame
                    mp_drawing.draw_landmarks(frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Extract the pose landmarks
                    landmarks = results.pose_landmarks.landmark
                    pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks])

                    # Append the pose data to the list
                    pose_data.append(pose_landmarks)

                    # Save the frame with the pose name and frame number
                    frame_path = os.path.join(pose_dir, f"{pose_name}_{frame_count}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

                    frame_count += 1

                # Convert the frame to PIL Image
                img = Image.fromarray(frame_rgb)

                # Convert the PIL Image to ImageTk format
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the webcam label with the new frame
                webcam_label.imgtk = imgtk
                webcam_label.configure(image=imgtk)

                # Update the GUI
                window.update()

            # Save the pose data as a NumPy array
            pose_data = np.array(pose_data)
            np.save(os.path.join(pose_dir, f"{pose_name}_data.npy"), pose_data)

            recording = False
            pose_entry.delete(0, tk.END)

    # Update the GUI
    window.update()

# Release the webcam
cap.release()
