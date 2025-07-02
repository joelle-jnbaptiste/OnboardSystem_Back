from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import requests
import tempfile
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PART_URLS = [
    "https://modelstorages.blob.core.windows.net/models/part_aa?sp=r&st=2025-06-27T08:54:15Z&se=2026-06-27T16:54:15Z&spr=https&sv=2024-11-04&sr=b&sig=dWLcmOxg3vrJeJFA5cbcuF1qoQOZ%2Bxhbhli0vOHHHRg%3D",
    "https://modelstorages.blob.core.windows.net/models/part_ab?sp=r&st=2025-06-27T08:55:11Z&se=2026-06-27T16:55:11Z&spr=https&sv=2024-11-04&sr=b&sig=GlUcrwfze3Xhi7eG2xJhJ5p%2BKTjophENAVUqnVx16Hk%3D",
    "https://modelstorages.blob.core.windows.net/models/part_ac?sp=r&st=2025-06-27T08:55:31Z&se=2026-06-27T16:55:31Z&spr=https&sv=2024-11-04&sr=b&sig=COULjlFjD%2BXZNf5zqmjMF2yLlc5t%2B9toWbBZuGjTp5c%3D",
    "https://modelstorages.blob.core.windows.net/models/part_ad?sp=r&st=2025-06-27T08:55:57Z&se=2026-06-27T16:55:57Z&spr=https&sv=2024-11-04&sr=b&sig=qxFKlQQo2B8xrGexu%2BsJ%2Bh2pS2rmwYX24C%2FZL84rhbw%3D",
    "https://modelstorages.blob.core.windows.net/models/part_ae?sp=r&st=2025-06-27T08:56:21Z&se=2026-06-27T16:56:21Z&spr=https&sv=2024-11-04&sr=b&sig=Q6elqQlD055oGJeLM%2FrHf2HjfHT6KqI0Ixeqo1BqX14%3D",
    "https://modelstorages.blob.core.windows.net/models/part_af?sp=r&st=2025-06-27T08:56:40Z&se=2026-06-27T16:56:40Z&spr=https&sv=2024-11-04&sr=b&sig=IlGMjqoYt%2F7vL6vuopX5zQIsMDR3GrpylK6BGu8nSPM%3D",
    "https://modelstorages.blob.core.windows.net/models/part_ag?sp=r&st=2025-06-27T08:57:00Z&se=2026-06-27T16:57:00Z&spr=https&sv=2024-11-04&sr=b&sig=TIObjYwwhNd315U73VciiZKwj7wisk%2ByYCwnqGw63rw%3D",
    "https://modelstorages.blob.core.windows.net/models/part_ah?sp=r&st=2025-06-27T08:57:21Z&se=2026-06-27T16:57:21Z&spr=https&sv=2024-11-04&sr=b&sig=eCzaw5CflaiIyvMcC71KK2n4VN3VG9tW4IqmPDY6xyA%3D",
    "https://modelstorages.blob.core.windows.net/models/part_ai?sp=r&st=2025-06-27T08:57:38Z&se=2026-06-27T16:57:38Z&spr=https&sv=2024-11-04&sr=b&sig=San3hyjZDyzcML4pDoB68jG9BoqVdGnBVgxJ4S5zzAw%3D",
    "https://modelstorages.blob.core.windows.net/models/part_aj?sp=r&st=2025-06-27T08:58:06Z&se=2026-06-27T16:58:06Z&spr=https&sv=2024-11-04&sr=b&sig=2%2BcZeOlXPTvIxPDVdWO4se%2Fpk%2FjZDkFKtRVvpTWDh3A%3D",
    "https://modelstorages.blob.core.windows.net/models/part_ak?sp=r&st=2025-06-27T08:58:27Z&se=2026-06-27T16:58:27Z&spr=https&sv=2024-11-04&sr=b&sig=HxLW5I%2BKmx18z4esbtIt8vF6WbtdcAiymIV51WURx9c%3D"
]


def download_and_merge_parts(urls):
    tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)

    for url in urls:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Erreur de téléchargement : {url}")
        tmp.write(response.content)

    tmp.close()
    return tmp.name


model = None

IMG_HEIGHT, IMG_WIDTH = 256, 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

app = FastAPI()


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


@app.on_event("startup")
async def load_model():
    model_path = download_and_merge_parts(PART_URLS)
    global model
    model = tf.keras.models.load_model(model_path)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = preprocess_image(file)
    prediction = model.predict(image)
    output_image = postprocess_prediction(prediction)
    return Response(content=output_image, media_type="image/png")
