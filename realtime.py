import cv2
from keras.models import model_from_json
import numpy as np

# Load pre-trained model
model_path = "D:/my_project/Emotion Detection Behind the Screen Sentiment Analysis of online Course/emotiondetector.h5"
model_json_path = "D:/my_project/Emotion Detection Behind the Screen Sentiment Analysis of online Course/emotiondetector.json"

# Load model architecture
with open(model_json_path, "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load model weights
model.load_weights(model_path)
print("Model loaded successfully!")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print(f"Error: Haar Cascade file not found at {haar_file}")
    exit()

# Function to preprocess input images for the model
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

print("Press 'q' to quit the application.")

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract and preprocess the face region
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face_features = extract_features(face)

        # Predict emotion
        prediction = model.predict(face_features)
        prediction_label = labels[np.argmax(prediction)]

        # Draw rectangle around the face and display emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
