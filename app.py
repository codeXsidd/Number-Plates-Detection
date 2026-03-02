import os
import cv2
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from skimage.transform import resize
import imutils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = pickle.load(open('finalized_model.sav', 'rb'))

if not os.path.exists('uploads'):
    os.makedirs('uploads')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['video']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    plate_number = process_video(filepath)

    return jsonify({'plate_number': plate_number})


# -------------------------
# IMPROVED VIDEO PROCESSING
# -------------------------

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    results = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 5th frame (faster + stable)
        if frame_count % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plate = detect_plate(gray)

        if plate is not None:
            plate_text = predict_characters(plate)
            if plate_text:
                results.append(plate_text)

    cap.release()

    if len(results) == 0:
        return "Plate Not Found"

    # Return most frequent prediction
    return max(set(results), key=results.count)


# -------------------------
# BETTER PLATE DETECTION
# -------------------------

def detect_plate(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 100, 200)

    contours, _ = cv2.findContours(edged.copy(),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Proper license plate filtering
        if 2 < aspect_ratio < 6 and w > 100 and h > 30:
            plate = gray[y:y+h, x:x+w]
            return plate

    return None


# -------------------------
# IMPROVED CHARACTER PREDICTION
# -------------------------

def predict_characters(plate):

    _, thresh = cv2.threshold(plate, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    characters = []
    positions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter character size
        if 20 < h < plate.shape[0] and 5 < w < plate.shape[1] / 2:
            roi = thresh[y:y+h, x:x+w]

            resized = resize(roi, (20, 20))
            resized = resized.reshape(1, -1)

            prediction = model.predict(resized)

            characters.append(prediction[0])
            positions.append(x)

    if len(characters) == 0:
        return None

    # Sort characters left to right
    sorted_chars = [char for _, char in sorted(zip(positions, characters))]

    plate_string = ''.join(sorted_chars)

    return plate_string


if __name__ == '__main__':
    app.run(debug=True)
