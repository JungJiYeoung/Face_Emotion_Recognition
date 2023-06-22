import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import subprocess

def execute_python_file(emotion):
    python_file = f"{emotion}.py"

    if os.path.exists(python_file):
        subprocess.call(["python", python_file])

# Define emotion categories
categories = ['happy', 'angry', 'neutral', 'sad', 'surprise']

# Load the trained model
model_path = 'C:/Users/wldud/Downloads/model/model/keypoint_classifier/keypoint_classifier.hdf5'
model = tf.keras.models.load_model(model_path)

# Set up Mediapipe modules for face detection and face mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Define function to extract facial landmarks
def extract_landmarks(image):
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            return [(landmark.x, landmark.y) for landmark in landmarks]

        return None

# Initialize webcam capture
picture_folder = './picture'
if os.path.exists(picture_folder):
    for root, dirs, files in os.walk(picture_folder):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

cap = cv2.VideoCapture(0)

# Create the picture directory if it doesn't exist
if not os.path.exists('./picture'):
    os.makedirs('./picture')

# Set up variables for capturing image after 10 seconds
capture_image = False
start_time = None

while True:
    ret, frame = cap.read()

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Extract facial landmarks from the frame
    landmarks = extract_landmarks(frame)

    if landmarks:
        # Normalize the landmarks' values
        landmarks = np.array(landmarks)
        landmarks = (landmarks - landmarks.min(axis=0)) / (landmarks.max(axis=0) - landmarks.min(axis=0))

        # Reshape the input to match the model's expected shape
        input_data = np.expand_dims(landmarks.flatten(), axis=0)

        # Make predictions using the loaded model
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions)
        predicted_emotion = categories[predicted_class_index]

        # Display the predicted emotion on the frame
        cv2.putText(frame, predicted_emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate remaining time
    if not capture_image:
        if start_time is None:
            start_time = time.time()

        elapsed_time = time.time() - start_time
        remaining_time = max(0, 5 - elapsed_time)

        # Display remaining time on the frame
        cv2.putText(frame, f"Time: {int(remaining_time)}s", (frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        if remaining_time <= 0:
            capture_image = True

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Capture the image
    if capture_image:
        captured_image = frame.copy()

        # Generate a unique filename based on the captured emotion and current timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"picture/{predicted_emotion}.jpg"
        cv2.imwrite(filename, captured_image)
        print("Image captured and saved as", filename)

        # Execute the Python file based on the captured emotion
        execute_python_file(predicted_emotion)

        break
    if cv2.waitKey(1) == 27:
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()