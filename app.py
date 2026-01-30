import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# تحميل الموديل
model = load_model("model/eye_disease_model.h5")

# أسماء الأصناف (نفس ترتيب التدريب)
class_names = [
    "Central Serous Chorioretinopathy",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa"
]

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    image_path = None
    prediction = None
    confidence = None

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(image_path)

                img = image.load_img(image_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array)[0]
                idx = np.argmax(preds)

                prediction = class_names[idx]
                confidence = round(preds[idx] * 100, 2)

    return render_template(
        "index.html",
        image_path=image_path,
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
