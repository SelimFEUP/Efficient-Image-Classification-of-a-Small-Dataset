import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import random
import librosa
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

path_mnetadata = './data/ESC-50-master/meta/esc50.csv'
path_audio = './data/ESC-50-master/audio/'

def load_metadata(path):
    df = pd.read_csv(path)
    df['fold'] = df['fold'].astype(str)  # Ensure fold is treated as a string
    return df

def load_audio_files(df, audio_path):
    audio_files = []
    labels = []
    for index, row in df.iterrows():
        file_path = audio_path + row['filename']
        audio_files.append(file_path)
        labels.append(row['category'])
    return audio_files, labels

data = load_metadata(path_mnetadata)
print(data.head())
audio_files, labels = load_audio_files(data, path_audio)
print(f"Loaded {len(audio_files)} audio files.")

# Use MFCC features for audio classification
def extract_mfcc(file_path, n_mfcc=13, max_pad_len=100, augment=False):
    audio, sr = librosa.load(file_path, sr=None)
    if augment:
        # Random time shift
        shift = np.random.randint(sr * 0.1)
        audio = np.roll(audio, shift)
        # Add random noise
        audio = audio + 0.005 * np.random.randn(len(audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.T  # Transpose to have time steps as rows

print("Extracting MFCC features...")
mfcc_features = []
for file in audio_files:
    mfcc = extract_mfcc(file)
    mfcc_features.append(mfcc)
mfcc_features = np.array(mfcc_features)
print(f"Extracted MFCC features shape: {mfcc_features.shape}")

# split the dataset into training, valid and testing sets
def split_dataset(features, labels, test_size=0.2, valid_size=0.2):
    X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=test_size, stratify=labels, random_state=random_seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size/(1-test_size), stratify=y_temp, random_state=random_seed)
    return X_train, X_valid, X_test, y_train, y_valid, y_test
X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(mfcc_features, labels)
print(f"Training set shape: {X_train.shape}, Validation set shape: {X_valid.shape}, Test set shape: {X_test.shape}")

# Convert labels to categorical
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Load YAMNet model from TF Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

def extract_yamnet_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)  # YAMNet expects 16kHz
    waveform = audio.astype(np.float32)
    # YAMNet expects mono waveform in [-1.0, 1.0]
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    # Run YAMNet and get embeddings (features)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    # Use mean embedding over time
    return np.mean(embeddings.numpy(), axis=0)

print("Extracting YAMNet features...")
yamnet_features = []
for file in audio_files:
    yamnet_features.append(extract_yamnet_features(file))
yamnet_features = np.array(yamnet_features)
print(f"Extracted YAMNet features shape: {yamnet_features.shape}")

def load_data():
    # Split dataset as before
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(yamnet_features, labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)
    return X_train, X_valid, X_test, y_train_encoded, y_valid_encoded, y_test_encoded, num_classes   
