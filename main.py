from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TFLITE_MODEL_PATH = "model/model_quant.tflite"

IMG_HEIGHT, IMG_WIDTH = 256, 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

app = FastAPI()
interpreter = None
input_details = None
output_details = None


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
    global interpreter, input_details, output_details
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = preprocess_image(file)

    input_data = image.astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_image = postprocess_prediction(output_data)
    return Response(content=output_image, media_type="image/png")
