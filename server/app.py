from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained models for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the age detection model
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

# Age groups for the age detector
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def is_baby(age_label):
    return age_label in ['(0-2)', '(4-6)']

def is_negative_emotion(emotion):
    negative_emotions = ['angry', 'disgust', 'fear', 'sad']
    return emotion in negative_emotions

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    logging.info(f"Received file: {image_file.filename}")
    np_img = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]

            # Prepare the face for age detection
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age_index = age_preds[0].argmax()
            age_label = AGE_GROUPS[age_index]

            logging.info(f"Detected age group: {age_label}")

            if is_baby(age_label):
                # Analyze the face using DeepFace for emotion detection
                result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                logging.info(f"DeepFace result: {result}")

                # Check if result is a list or dictionary
                if isinstance(result, list):
                    if len(result) > 0 and 'dominant_emotion' in result[0]:
                        emotion = result[0]['dominant_emotion']
                    else:
                        return jsonify({'error': 'Emotion detection failed'}), 500
                elif isinstance(result, dict):
                    emotion = result.get('dominant_emotion', 'unknown')
                else:
                    return jsonify({'error': 'Unexpected result format from DeepFace'}), 500

                logging.info(f"Detected emotion: {emotion}")

                # Check for negative emotion
                if is_negative_emotion(emotion):
                    return jsonify({'emotion': emotion, 'notification': 'Negative emotion detected'})
                return jsonify({'emotion': emotion})

        return jsonify({'error': 'No baby face detected'}), 400

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)