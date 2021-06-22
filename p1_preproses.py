import pickle
import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

DIRECTORY = './Dataset2/latih/'
CATEGORIES = ['H2O', 'H2O+NaOH']

data = []
img_size = 100
im_arr = []

for category in CATEGORIES:
    count = 0
    path = os.path.join(DIRECTORY, category)
    for pic in os.listdir(path):
        img_path = os.path.join(path, pic)
        label = CATEGORIES.index(category)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        data.append([img, label])
        count = count + 1
        if(count <= 10):  # Menjadi 20 Buah, karena 2 direktori atau 2 kali diulang baca direktori
            im_arr.append({category: img})
    print("Jumlah "+str(category)+" : "+str(count))

random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

fig, axs = plt.subplots(4, 5, figsize=(20, 10))
cnt = 0
row = 0
col = 0
for i in im_arr:
    for key, value in i.items():
        if(cnt == 5):
            row = row+1
            col = 0
            cnt = 0
        axs[row, col].imshow(value)
        axs[row, col].set_title(key)
        cnt = cnt+1
        col = col+1

plt.show()

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))
