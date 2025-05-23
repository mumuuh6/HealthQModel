from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model once when the file is imported
model = load_model("model/melanoma_model.h5")

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Update to your model's input size
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape((1, 224, 224, 3))
    return image_array

def predict_melanoma(image: Image.Image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    result = "Malignant" if prediction > 0.5 else "Benign"
    confidence = float(prediction) if result == "Malignant" else 1 - float(prediction)
    return {"result": result, "confidence": round(confidence, 2)}
