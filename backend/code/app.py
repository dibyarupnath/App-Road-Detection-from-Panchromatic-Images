from flask import Flask, request, render_template, redirect, url_for
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from roadseg_nn import RoadSegNN
from segnet import SegNet
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import subprocess


app = Flask(__name__, template_folder=os.path.join(
    os.path.dirname(__file__), '../../templates'), static_folder=os.path.join(os.path.dirname(__file__), '../../static'))


@app.route('/')
def index():
    return render_template('road_detector.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected model type
    model_type = request.form['model_type']

    # Get the sat image
    img_path = request.form['img_path']

    # Run the Python script and pass form data
    subprocess.run(['python', 'detect_roads.py', model_type, img_path])

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
