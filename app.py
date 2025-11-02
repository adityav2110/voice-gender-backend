from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import os
from flask_cors import CORS

app = Flask(__name__)

# ====== CORS SETUP ======
frontend_origin = os.getenv("FRONTEND_URL", "http://localhost:3000")
CORS(app, origins=[frontend_origin])

# ====== MODEL CONSTANTS ======
TARGET_SR = 22050
DURATION = 4
N_MFCC = 40
HOP_LENGTH = 512
N_FRAMES = 173
FIXED_LENGTH = int(DURATION * TARGET_SR)
INPUT_SHAPE = (1, N_MFCC, N_FRAMES, 1)

# ====== MODEL PATHS ======
keras_model_path = "model.keras"
tflite_model_path = "model.tflite"

# ====== AUTO-CONVERT KERAS → TFLITE ======
if not os.path.exists(tflite_model_path) and os.path.exists(keras_model_path):
    print("🔄 Converting model.keras → model.tflite ...")
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print("✅ Conversion complete: model.tflite created!")

# ====== LOAD TFLITE MODEL ======
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ====== PREPROCESSING FUNCTION ======
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) > FIXED_LENGTH:
            start = (len(y) - FIXED_LENGTH) // 2
            y = y[start : start + FIXED_LENGTH]
        elif len(y) < FIXED_LENGTH:
            y = np.pad(y, (0, FIXED_LENGTH - len(y)), 'constant')
        mfcc = librosa.feature.mfcc(
            y=y, sr=TARGET_SR, n_mfcc=N_MFCC, n_fft=2048, hop_length=HOP_LENGTH
        )
        if mfcc.shape != (N_MFCC, N_FRAMES):
            mfcc = librosa.util.fix_length(mfcc, size=N_FRAMES, axis=1)
        mfcc = mfcc.reshape(1, N_MFCC, N_FRAMES, 1).astype(np.float32)
        return mfcc
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {e}")

# ====== ROUTES ======
@app.route('/')
def home():
    return "✅ Voice Gender Recogniser Backend is Running"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file') or request.files.get('audio')
    if file is None or file.filename == '':
        return jsonify({'error': 'No audio file received'}), 400

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    try:
        mfcc = preprocess_audio(file_path)
        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        probability = float(prediction[0][0])
        if probability > 0.5:
            label = "Male"
            confidence = probability * 100
        else:
            label = "Female"
            confidence = (1 - probability) * 100
        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ====== RUN SERVER ======
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
