from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import requests
import tempfile
import os


PART_URLS = [
    "sp=r&st=2025-06-27T07:15:13Z&se=2026-06-27T15:15:13Z&spr=https&sv=2024-11-04&sr=b&sig=3v9R9xVOQ2xjiW8MILUDNWiGjEmZlXeodaSjf6ZXzd0%3D",
    "sp=r&st=2025-06-27T07:16:30Z&se=2026-06-27T15:16:30Z&spr=https&sv=2024-11-04&sr=b&sig=L0KT9xymqrSkC%2FJL03ZdrnTVb1uZy1EGUrObgxmjQSo%3D",
    "sp=r&st=2025-06-27T07:16:48Z&se=2026-06-27T15:16:48Z&spr=https&sv=2024-11-04&sr=b&sig=Ar0wEq6meenllBVoEiUW2UZqiupmTlGP1Oly1QbUP%2BQ%3D",
    "sp=r&st=2025-06-27T07:17:40Z&se=2026-06-27T15:17:40Z&spr=https&sv=2024-11-04&sr=b&sig=Slkb7p6foP%2FHl2qCZgWJFaXHIrZ10U0k1NEWUUGOVeU%3D",
    "sp=r&st=2025-06-27T07:17:55Z&se=2025-06-27T15:17:55Z&spr=https&sv=2024-11-04&sr=b&sig=brisFh4qeCZDhMCxiMzbas7bv8SlvB6Q11iOgR2BaNk%3D",
    "sp=r&st=2025-06-27T07:18:12Z&se=2026-06-27T15:18:12Z&spr=https&sv=2024-11-04&sr=b&sig=rMOT%2BjTo6lkbK3fb9mXfhslOkRZndUu418MT%2BLk%2Fqf4%3D",
    "sp=r&st=2025-06-27T07:18:29Z&se=2026-06-27T15:18:29Z&spr=https&sv=2024-11-04&sr=b&sig=CGgZ1hT6xpQGBur6A7eDL0zNwHkMDUJESzXwzZseN1I%3D",
    "sp=r&st=2025-06-27T07:18:46Z&se=2026-06-27T15:18:46Z&spr=https&sv=2024-11-04&sr=b&sig=%2BbqJvvTadVrzhHRDS1%2Fss6h9WNK%2ByVh48QYYuxkA78s%3D",
    "sp=r&st=2025-06-27T07:19:07Z&se=2026-06-27T15:19:07Z&spr=https&sv=2024-11-04&sr=b&sig=N%2BZWEDKNHICU3KEU%2BNKwzf2l0oKT7wsxLwI1yFldtNs%3D",
    "sp=r&st=2025-06-27T07:19:25Z&se=2026-06-27T15:19:25Z&spr=https&sv=2024-11-04&sr=b&sig=JB%2F%2FzEFLZQRqg7TG1IvGy2547WpxzQccL41w5OqHS00%3D",
    "sp=r&st=2025-06-27T07:19:47Z&se=2026-06-27T15:19:47Z&spr=https&sv=2024-11-04&sr=b&sig=CWaPvNRSF06JDzuC56gme249NQUAATw86omtSufd%2BZI%3D"
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
