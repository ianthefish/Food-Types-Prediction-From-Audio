import os
import numpy as np
import librosa
import hashlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

#function to get unique file hash
def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

#function to extract Mel Spectrogram features
def extract_features(file_path, max_pad_len=100):
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    pad_width = max_pad_len - mel_spec.shape[1]
    if pad_width > 0:
        mel_spec = np.pad(mel_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec = mel_spec[:, :max_pad_len]
    return mel_spec

#function to extract MFCC features
def extract_mfcc(file_path, max_pad_len=100, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.flatten()

#function to apply augmentation to an audio signal
def augment_audio(y, sr):
    #randomly apply time stretching, pitch shifting, and noise addition
    if random.random() > 0.5:
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))
    if random.random() > 0.5:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.randint(-3, 3))
    if random.random() > 0.5:
        y = y + 0.005 * np.random.randn(len(y))
    return np.clip(y, -1, 1)

# Load dataset
X, y = [], []
hashes = {}


dataset_path = "./dataset"
#decide whether to apply augmentation or not
apply_augmentation = 1  

for label in sorted(os.listdir(dataset_path)):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                file_hash = get_file_hash(file_path)
                if file_hash not in hashes:
                    hashes[file_hash] = file_path
                    #extract features
                    X.append(extract_features(file_path))
                    # X.append(extract_mfcc(file_path))
                    y.append(label)

X = np.array(X)
y = np.array(y)

#encode labels because the model only accepts numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X = X[..., np.newaxis] 

#normalize features
scaler = StandardScaler()
X = X.reshape(X.shape[0], -1)
X = scaler.fit_transform(X)


X = X.reshape(-1, 40, 100, 1)

#define CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(256, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#cross-validation setup
num_classes = len(set(y_encoded))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_accuracies = []
final_cm = np.zeros((num_classes, num_classes)) 

#declare lists to store history of each fold
history_acc = []
history_loss = []
history_val_acc = []
history_val_loss = []

#perform cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"Training Fold {fold+1}/5")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]


    #apply augmentation
    if apply_augmentation:
        X_train_augmented = []
        for i in range(len(train_idx)):
            original_file_path = hashes.get(y[train_idx[i]])
            if original_file_path:
                y_audio, sr = librosa.load(original_file_path, sr=None)
                #use augment_audio function to augment audio
                y_audio = augment_audio(y_audio, sr)
                #extract features
                X_train_augmented.append(extract_features(y_audio))
                # X_train_augmented.append(extract_mfcc(y_audio))
            else:
                X_train_augmented.append(X_train[i, :, :, 0])

        X_train_augmented = np.array(X_train_augmented)
        X_train_augmented = X_train_augmented[..., np.newaxis] 
        X_train = np.concatenate((X_train, X_train_augmented), axis=0)
        y_train = np.concatenate((y_train, y_train), axis=0)

    model = create_cnn_model((40, 100, 1), num_classes)

    #train the model and capture the history
    history = model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test), verbose=0)

    #store accuracy and loss for each fold
    history_acc.append(history.history['accuracy'])
    history_loss.append(history.history['loss'])
    history_val_acc.append(history.history['val_accuracy'])
    history_val_loss.append(history.history['val_loss'])

    #evaluate model on the test data
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    cross_val_accuracies.append(acc)
    print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    #compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    final_cm += cm  # Sum up confusion matrices from all folds

#final evaluation
avg_accuracy = np.mean(cross_val_accuracies)
print(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f}")

#normalize the confusion matrix (percentage)
final_cm_normalized = final_cm.astype('float') / final_cm.sum(axis=1)[:, np.newaxis]

#plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(final_cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

#plot accuracy and loss for each fold
epochs = range(1, 61)  #6D0 epochs

plt.figure(figsize=(12, 6))
for fold in range(5):
    plt.plot(epochs, history_acc[fold], label=f'Fold {fold+1} Train Accuracy')
    plt.plot(epochs, history_val_acc[fold], label=f'Fold {fold+1} Val Accuracy', linestyle='--')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(12, 6))
for fold in range(5):
    plt.plot(epochs, history_loss[fold], label=f'Fold {fold+1} Train Loss')
    plt.plot(epochs, history_val_loss[fold], label=f'Fold {fold+1} Val Loss', linestyle='--')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
