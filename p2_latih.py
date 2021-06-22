import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD

X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_val /= 255
y_train = to_categorical(y_train, 2)  # [1,0]=0, [0,1]=1
y_val = to_categorical(y_val, 2)


model = Sequential()  # model = sequential
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
          input_shape=(100, 100, 3)))  # layer convolutional 2D
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling with stride (2,2)
model.add(Conv2D(32, (3, 3), activation='relu'))  # layer convolutional 2D
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling with stride (2,2)
model.add(Dropout(0.25))  # delete neuron randomly while training and remain 75%
model.add(Flatten())  # make layer flatten
model.add(Dense(128, activation='relu'))  # fully connected layer
model.add(Dropout(0.5))  # delete neuron randomly and remain 50%
model.add(Dense(2, activation='softmax'))  # softmax works

# Melatih dataset dengan CNN
epochs = 20
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
print(model.summary())

my_model = model.fit(X_train, y_train, validation_data=(
    X_val, y_val), epochs=epochs, batch_size=32)
scores = model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('my_model.h5')

# Plot Train and validation accuracy
acc = my_model.history['accuracy']
val_acc = my_model.history['val_accuracy']
loss = my_model.history['loss']
val_loss = my_model.history['val_loss']
epochs = range(1, len(acc) + 1)
# Train and validation accuracy
plt.figure()
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accurarcy')
plt.legend()
# Train and validation loss
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

# Prediksi
y_pred = model.predict(X_val)
y_pred_class = np.argmax(y_pred, axis=1)
y_val_class = np.argmax(y_val, axis=1)
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
