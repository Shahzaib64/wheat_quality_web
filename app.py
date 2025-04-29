from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)
model = tf.keras.models.load_model("mlp_model.h5")

labels = ['damaged_grain', 'foreign', 'grain', 'broken_grain', 'grain_cover']

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    language = request.args.get("lang", "en")

    if request.method == "POST":
        image = request.files["image"]
        if image.filename != "":
            filename = f"{uuid.uuid4().hex}_{image.filename}"
            img_path = os.path.join("static", filename)
            image.save(img_path)
            image_path = img_path

            img = Image.open(img_path).convert('RGB').resize((100, 100))
            img_array = np.array(img)

            meanR = np.mean(img_array[:, :, 0])
            meanG = np.mean(img_array[:, :, 1])
            meanB = np.mean(img_array[:, :, 2])

            # Geometry-based features
            gray_img = img.convert('L')
            thresh = np.array(gray_img) > 128
            area = np.sum(thresh)
            perimeter = np.sum(thresh[:, :-1] != thresh[:, 1:]) + np.sum(thresh[:-1, :] != thresh[1:, :])

            eig1 = np.max(img_array.shape)
            eig2 = np.min(img_array.shape)
            eccentricity = np.sqrt(1 - (eig2 / eig1)**2)

            features = np.array([[area, perimeter, meanR, meanG, meanB, eig1, eig2, eccentricity]])

            predictions = model.predict(features)
            predicted_index = np.argmax(predictions)

            result = {label: 1 if i == predicted_index else 0 for i, label in enumerate(labels)}

    return render_template("index.html", result=result, image=image_path, language=language)

if __name__ == "__main__":
    app.run(debug=True)
