from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import requests

app = FastAPI()

# Google Drive File ID
file_id = "17kbJOY8glURd0iNAoESlokg6i2FJPjU4"
file_name = "xception.keras"
drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Download model if not exists
if not os.path.exists(file_name):
    print("🔽 Downloading model from Google Drive...")
    response = requests.get(drive_url)
    with open(file_name, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded.")

# Load model
try:
    model = tf.keras.models.load_model(file_name)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model loading failed:", e)

image_size = 256
labels = ["Not Melanoma", "Melanoma"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((image_size, image_size))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return JSONResponse({
            "prediction": labels[predicted_class],
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
