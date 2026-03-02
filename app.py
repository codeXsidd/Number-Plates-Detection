from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import pickle
import SegmentCharacters
import PredictCharacters

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "License Plate Recognition API Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Here you call your prediction logic
    os.system("python PredictCharacters.py")

    return jsonify({"message": "Prediction completed. Check logs."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
