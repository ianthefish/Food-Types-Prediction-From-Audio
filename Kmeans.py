import os
import numpy as np
import librosa
import hashlib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment



#function to get unique file hash
def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

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
                    # X.append(extract_mfcc(file_path))
                    X.append(extract_features(file_path))
                    y.append(label)
                    
                    #apply augmentation
                    if apply_augmentation:
                        y_audio, sr = librosa.load(file_path, sr=None)
                        #use augmented audio to extract features
                        y_aug = augment_audio(y_audio, sr)
                        # X.append(extract_mfcc(file_path))
                        X.append(extract_features(file_path))
                        y.append(label)

X = np.array(X)
y = np.array(y)

#encode labels because the model only accepts numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#normalize features
scaler = StandardScaler()
X = X.reshape(X.shape[0], -1)
X = scaler.fit_transform(X)

#define clustering model and fit the data
num_clusters = len(set(y_encoded))
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)
y_pred = kmeans.predict(X)

#ues nmi homogeneity and ari to evaluate the clustering performance
nmi = normalized_mutual_info_score(y_encoded, y_pred)
homogeneity = homogeneity_score(y_encoded, y_pred)
ari = adjusted_rand_score(y_encoded, y_pred)

print(f"NMI: {nmi:.4f}, Homogeneity: {homogeneity:.4f}, ARI: {ari:.4f}")

#initialize a confusion matrix of shape (num_clusters, number of unique labels)
conf_matrix = np.zeros((num_clusters, len(set(y_encoded))))

#populate the confusion matrix: count occurrences of predicted vs true labels
for i in range(len(y_encoded)):
    conf_matrix[y_pred[i], y_encoded[i]] += 1

#use linear sum assignment to find the best label mapping
row_ind, col_ind = linear_sum_assignment(conf_matrix, maximize=True)
y_pred_matched = np.copy(y_pred)
for i in range(len(y_pred)):
    y_pred_matched[i] = col_ind[y_pred[i]]

conf_matrix_matched = np.zeros((num_clusters, len(set(y_encoded))))
for i in range(len(y_encoded)):
    conf_matrix_matched[y_pred_matched[i], y_encoded[i]] += 1


#plot the adjusted confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_matched, annot=True, fmt='g', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (K-Means Clustering with Matched Labels)")
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.show()
