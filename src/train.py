import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import random
from src.preprocessing import load_data, yamnet_features
from src.model import build_classifier
 
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

X_train, X_valid, X_test, y_train_encoded, y_valid_encoded, y_test_encoded, num_classes = load_data()

input_shape = (yamnet_features.shape[1],)
model = build_classifier(input_shape, num_classes)
model.summary()

# Train
def train_model():
    mc = tf.keras.callbacks.ModelCheckpoint('./models/best_yamnet_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train_encoded, validation_data=(X_valid, y_valid_encoded), epochs=100, batch_size=16, callbacks=[mc, early_stopping])
    return history
