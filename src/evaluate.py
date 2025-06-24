import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from src.train import model
from src.preprocessing import load_data, yamnet_features, extract_yamnet_features

X_train, X_valid, X_test, y_train_encoded, y_valid_encoded, y_test_encoded, num_classes = load_data()

model.load_weights('./models/best_yamnet_model.keras')

# Predict on a new audio file
def classify_audio_file_yamnet(file_path, model, label_encoder):
    features = extract_yamnet_features(file_path)
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    probabilities = predictions[0]
    return predicted_label, probabilities


