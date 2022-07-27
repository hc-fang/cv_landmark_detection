import numpy as np
import os
import cv2
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

image_filenames, landmarks = None, None
with open(os.path.join('../cv_data/data/synthetics_train/annot.pkl'), 'rb') as f:
    annot = pickle.load(f)
    image_filenames, landmarks = annot

landmarks = np.array(landmarks)
landmarks = landmarks.reshape(-1, 136)
print(landmarks.shape)
pca = PCA(n_components=1)
pca.fit(landmarks)
trans_landmarks = pca.transform(landmarks).flatten()
print(trans_landmarks.shape)

mean = np.mean(trans_landmarks)
std = np.std(trans_landmarks)

plt.hist(trans_landmarks, range = (-600, 600))
plt.savefig('./distribution_train.png')
plt.show()

# for idx in range(100):
#     tp = 'out'
#     if trans_landmarks[idx] > mean + std: tp = 'large'
#     elif trans_landmarks[idx] < mean - std: tp = 'small'
#     else: tp = 'mean'
#     image = cv2.imread(os.path.join('../cv_data/data/synthetics_train', image_filenames[idx]))
#     plt.figure(figsize=(10, 10))

#     plt.imshow(image)

#     plt.text(10, 10, trans_landmarks[idx], fontsize=20, color='white')
 
#     plt.savefig(os.path.join(f'./image/{tp}', image_filenames[idx]))
#     plt.close()

