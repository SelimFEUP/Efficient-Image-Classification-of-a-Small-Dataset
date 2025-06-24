import numpy as np
from src.preprocessing import load_data, label_encoder
from src.evaluate import classify_audio_file_yamnet, model
from src.train import train_model
from sklearn.metrics import classification_report

X_train, X_valid, X_test, y_train_encoded, y_valid_encoded, y_test_encoded, num_classes = load_data()

# Train
train_model()

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_))

file_path = './data/traffic-sound-111442.mp3'
predicted_label, probabilities = classify_audio_file_yamnet(file_path, model, label_encoder)
top_indices = np.argsort(probabilities)[-3:][::-1]
print(f"Predicted label: {predicted_label}")
print("Top 3 probabilities:")
for idx in top_indices:
    print(f"{label_encoder.classes_[idx]}: {probabilities[idx]:.4f}")

