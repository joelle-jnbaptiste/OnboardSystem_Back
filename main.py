from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import requests
import tempfile

url = "https://modelstorages.blob.core.windows.net/models/saved_model_keras_larger.keras?sp=r&st=2025-06-26T13:41:30Z&se=2026-02-28T22:41:30Z&spr=https&sv=2024-11-04&sr=b&sig=MkehvHbN%2B1H94JdLgsRfZmR8LicBwBM9ban4g%2B9kPg8%3D"

def load_model_from_blob(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        tmp.write(response.content)
        tmp.flush()
        model = tf.keras.models.load_model(tmp.name)
    return model

model = load_model_from_blob(url)

IMG_HEIGHT, IMG_WIDTH = 256, 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

app = FastAPI()

# model = tf.keras.models.load_model("model/saved_model_keras_larger.keras")


def preprocess_image(file: UploadFile) -> np.ndarray:
    image = Image.open(file.file).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    image = np.array(image).astype("float32") / 255.0
    image = (image - MEAN) / STD
    image = np.expand_dims(image, axis=0)

    return image


def postprocess_prediction(pred: np.ndarray) -> bytes:
    pred = np.squeeze(pred, axis=0)
    pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)

    color_palette = {
        0: (120, 120, 120),    # route
        1: (230, 100, 130),    # humain
        2: (70, 140, 210),     # voiture
        3: (170, 120, 190),    # bâtiment
        4: (255, 170, 100),    # objet
        5: (80, 200, 120),     # nature
        6: (100, 180, 230),    # ciel
        7: (220, 220, 220),    # blanc
    }

    h, w = pred_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in color_palette.items():
        color_img[pred_mask == class_idx] = color

    pil_img = Image.fromarray(color_img)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = preprocess_image(file)
    prediction = model.predict(image)
    output_image = postprocess_prediction(prediction)
    return Response(content=output_image, media_type="image/png")
