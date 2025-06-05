
# FastAPI Image Segmentation API

This project provides a simple FastAPI backend for image segmentation using a pretrained Keras model. The API accepts an image, resizes and normalizes it, runs inference, and returns a colorized segmentation mask.

---

## 🚀 Features

- Upload an image via POST request
- Automatic preprocessing (resize to 256×512 and normalization)
- Model prediction using a `.keras` model
- Postprocessing: `argmax` + color mapping per class
- Returns a colorized PNG mask

---

## 🧱 Project Structure

```
ONBOARDSYSTEM_BACK/
│
├── main.py             # FastAPI app
├── model/
│   └── your_model.keras
├── env/                # Python virtual environment (optional)
└── __pycache__/        # Bytecode cache (optional)
```

---

## ⚙️ Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv env
source env/bin/activate       # On Linux/macOS
env\Scripts\activate          # On Windows
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn pillow numpy tensorflow
```

---

## 🧪 Running the API

Start the API server:

```bash
uvicorn main:app --reload
```

The API will be available at:  
📍 `http://127.0.0.1:8000`

Docs available at:  
📄 `http://127.0.0.1:8000/docs`

---

## 📬 API Endpoint

### `POST /predict`

**Description:**  
Upload an image and get back the predicted segmentation mask as a PNG.

**Form-data parameters:**

| Key  | Type  | Description              |
|------|-------|--------------------------|
| file | image | Your input image (RGB)   |

**Response:**  
Returns a PNG image (segmentation mask).

---

## 🧠 Notes

- Input images are resized to 256x512.
- Normalization is done with ImageNet statistics.
- Output mask contains 8 semantic classes:
  - 0: road (gray)
  - 1: human (rose)
  - 2: car (blue)
  - 3: building (purple)
  - 4: object (orange)
  - 5: vegetation (green)
  - 6: sky (sky blue)
  - 7: white (light gray)

---

