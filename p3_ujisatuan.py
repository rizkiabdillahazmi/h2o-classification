import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

DIRECTORY = r'./Dataset2/uji/H2O/H2O 151.png'
CATEGORIES = ['H2O', 'H2O+NaOH']


def image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (100, 100))
    img = np.array(img)
    img = img.astype('float32')
    img /= 255
    return img


model = keras.models.load_model('my_model.h5')

img_uji = image(DIRECTORY)
X_uji = img_uji.reshape(-1, 100, 100, 3)
prediction = model.predict(X_uji)
accuracy = np.max(prediction) * 100
label = 'Hasil Prediksi : {:.2f}% {}'.format(
    accuracy, CATEGORIES[prediction.argmax()])

plt.figure()
plt.imshow(img_uji)
plt.title(label)

plt.show()
