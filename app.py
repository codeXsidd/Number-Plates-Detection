import os
import cv2
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
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


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    plate_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = imutils.rotate(gray, 270)

        threshold_value = threshold_otsu(gray)
        binary = gray > threshold_value

        label_image = measure.label(binary)

        for region in regionprops(label_image):
            if region.area < 50:
                continue

            min_row, min_col, max_row, max_col = region.bbox
            plate = binary[min_row:max_row, min_col:max_col]

            plate_result = predict_characters(plate)

            if plate_result:
                cap.release()
                return plate_result

    cap.release()
    return "Plate Not Found"


def predict_characters(plate):
    labelled = measure.label(np.invert(plate))

    characters = []
    column_list = []

    for regions in regionprops(labelled):
        y0, x0, y1, x1 = regions.bbox
        roi = plate[y0:y1, x0:x1]

        resized_char = resize(roi, (20, 20))
        characters.append(resized_char.reshape(1, -1))
        column_list.append(x0)

    result = []
    for char in characters:
        prediction = model.predict(char)
        result.append(prediction[0])

    plate_string = ''.join(result)
    return plate_string


if __name__ == '__main__':
    app.run(debug=True)
