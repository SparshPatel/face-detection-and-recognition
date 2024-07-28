import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Device list:", tf.config.list_physical_devices())
import numpy as np
import urllib.request
import os
# URL of the image to be processed
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Kulusuk%2C_Inuit_man_%286822268117%29.jpg/201px-Kulusuk%2C_Inuit_man_%286822268117%29.jpg"
image_filename = "downloaded_image.jpg"

# Download the image
if not os.path.exists(image_filename):
    urllib.request.urlretrieve(image_url, image_filename)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load a pre-trained model (e.g., ResNet50) for feature extraction
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

# Function to preprocess input image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Function to extract features using the pre-trained model
def extract_features(image):
    image = preprocess_image(image)
    features = model.predict(image)
    return features

# Simulated database of known faces (for demo purposes)
known_faces = []
known_labels = []

def load_known_faces():
    # Load known faces and labels (for demo, using random data)
    for i in range(5):
        known_faces.append(np.random.rand(2048))  # Example feature vector size from ResNet
        known_labels.append(f"Person {i+1}")

load_known_faces()

def recognize_face(features):
    # Simple recognition by comparing with known faces
    min_distance = float('inf')
    label = "Unknown"
    for known_feature, known_label in zip(known_faces, known_labels):
        distance = np.linalg.norm(features - known_feature)
        if distance < min_distance:
            min_distance = distance
            label = known_label
    return label

def detect_and_recognize_faces(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        features = extract_features(face_region)
        label = recognize_face(features[0])
        
        # Draw rectangle around the face and label it
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

# Test the function with the downloaded image
detect_and_recognize_faces(image_filename, "output_image.jpg")
