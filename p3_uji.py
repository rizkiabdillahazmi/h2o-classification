import random
import numpy as np
import os
import cv2
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

DIRECTORY = './Dataset2/uji/'
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
    print("Jumlah "+str(category)+" : "+str(count))

random.shuffle(data)

X_uji = []
y_uji = []

for features, label in data:
    X_uji.append(features)
    y_uji.append(label)

X_uji = np.array(X_uji)
y_uji = np.array(y_uji)

X_uji = X_uji.astype('float32')
X_uji /= 255
y_uji = to_categorical(y_uji, 2)

model = keras.models.load_model("my_model.h5")

# Prediksi
y_pred = model.predict(X_uji)
y_pred_class = np.argmax(y_pred, axis=1)
y_val_class = np.argmax(y_uji, axis=1)
# Report
print(classification_report(y_val_class, y_pred_class,
      target_names=["H2O", "H2O+NaOH"]))
print(accuracy_score(y_val_class, y_pred_class))
# heatmap
cm = confusion_matrix(y_val_class, y_pred_class)
plt.figure()
sn.heatmap(cm, annot=True, xticklabels=[
           "H2O", "H2O+NaOH"], yticklabels=["H2O", "H2O+NaOH"])

plt.show()
