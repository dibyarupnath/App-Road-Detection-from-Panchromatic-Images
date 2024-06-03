from flask import Flask, request, render_template, redirect, url_for
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import subprocess
import sys
import importlib.util
from detect_roads import detect_rd


def display_image(image_path):
    try:
        # Open an image file
        image = Image.open(image_path)
        # Display image
        image.show()
    except IOError:
        print("Unable to open image file.")


app = Flask(__name__, template_folder=os.path.join(
    os.path.dirname(__file__), 'templates'), static_folder=os.path.join(os.path.dirname(__file__), 'static'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detector')
def detector():
    return render_template('road_detector.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected model type
    model_type = request.form['model_type']

    # Get the sat image
    img_path = request.form['img_path']

    print(model_type, img_path)

    # subprocess.run(['python', 'detect_roads.py', model_type, img_path])
    output = detect_rd(model_type, img_path)

    display_image(output)

    return redirect(url_for('detector'))


if __name__ == '__main__':
    app.run(debug=True)
