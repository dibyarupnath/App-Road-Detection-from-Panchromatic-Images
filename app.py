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

    # Open an image file
    print("HAALLOOO", image_path)
    image = Image.open(image_path)
    # Display image
    image.show()


app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detector")
def detector():
    # img_path = "static\\blank.png"
    img_path = os.path.join("static", "blank.png")
    model_type_output = "None"
    input_img_name = "None"
    # output = "static\\blank.png"
    output = os.path.join("static", "blank.png")
    return render_template(
        "road_detector.html",
        input_img_path=img_path,
        model_type_output=model_type_output,
        input_img_name=input_img_name,
        output_img=output,
    )
    # return render_template('road_detector.html')


@app.route("/predict", methods=["POST"])
def predict():
    model_type_output = ""

    # Get the selected model type
    model_type = request.form["model_type"]

    if model_type == "Blank":
        # img_path = "static\\blank.png"
        img_path = os.path.join("static", "blank.png")
        model_type_output = "None"
        input_img_name = "None"
        # output = "static\\blank.png"
        output = os.path.join("static", "blank.png")
        return render_template(
            "road_detector.html",
            input_img_path=img_path,
            model_type_output=model_type_output,
            input_img_name=input_img_name,
            output_img=output,
        )

    # Get the INPUT Sat image

    # Saving the input-image-name to return to the frontend at the end of the function
    input_img_name = request.form["img_path"]

    # img_path = request.form['img_path']
    img_path = os.path.join(".", "static", "test_data",
                            request.form["img_path"])

    print(model_type, img_path)

    # subprocess.run(['python', 'detect_roads.py', model_type, img_path])
    output = detect_rd(model_type, img_path)

    # Returning Model Type
    # img_path = img_path.replace('\\', '\\\\')
    if model_type == "ResNet-50":
        model_type_output = "RoadSegNN with the ResNet-50 backbone"

    elif model_type == "ResNet-101":
        model_type_output = "RoadSegNN with the ResNet-101 backbone"

    elif model_type == "Swin-T":
        model_type_output = "RoadSegNN with the Swin-T backbone"

    elif model_type == "SegNet":
        model_type_output = "SegNet"

    return render_template(
        "road_detector.html",
        input_img_path=img_path,
        model_type_output=model_type_output,
        input_img_name=input_img_name,
        output_img=output,
    )


if __name__ == '__main__':
    app.run(debug=True)
