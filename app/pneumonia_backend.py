from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Pneumonia Prediction API is Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
