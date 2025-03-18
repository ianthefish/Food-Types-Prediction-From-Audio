import os
import numpy as np
import librosa
import hashlib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
    #return mel_spec as 1D array
    return mel_spec.flatten()

#function to extract MFCC features
def extract_mfcc(file_path, max_pad_len=100, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    #return mfcc as 1D array
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

#normalize features
scaler = StandardScaler()
X = X.reshape(X.shape[0], -1)  # Flatten to 2D for scaling
X = scaler.fit_transform(X)

#cross-validation setup
num_classes = len(set(y_encoded))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_accuracies = []
final_cm = np.zeros((num_classes, num_classes))


#perform cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"Training Fold {fold+1}/5")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    #apply augmentation
    if apply_augmentation:
        X_train_augmented = []
        y_train_augmented = []

        for i in range(len(train_idx)):
            original_file_path = hashes.get(y[train_idx[i]])
            if original_file_path:
                y_audio, sr = librosa.load(original_file_path, sr=None)
                #use augment_audio function to augment audio
                y_audio = augment_audio(y_audio, sr)
                #extract features
                X_train_augmented.append(extract_features(original_file_path))
                # X_train_augmented.append(extract_mfcc(original_file_path))
                y_train_augmented.append(y_train[i])

        X_train_augmented = np.array(X_train_augmented)
        y_train_augmented = np.array(y_train_augmented)

        X_train = np.concatenate((X_train, X_train_augmented), axis=0)
        y_train = np.concatenate((y_train, y_train_augmented), axis=0)


    #train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #evaluate model on the test data
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cross_val_accuracies.append(acc)
    print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    #compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    final_cm += cm  #sum up confusion matrices from all folds

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
