import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from tensorflow import keras
from keras.layers import Dropout, Flatten
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# Parâmetros
path_visual = "Imagens"
path_text = "Imagens"
batch_size_val = 10
steps_per_epoch_val = 20
epochs_val = 20
imageDimesions = (32, 32, 3)

# Funções de pré-processamento e carregamento de imagens de texto
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def load_and_process_text(path):
    text_images = []
    text_classNo = []

    pastas = os.listdir(path)
    print("Total de Classes de Texto:", len(pastas))
    noOfClasses = len(pastas)

    for pt in range(0, len(pastas)):
        arquivos = os.listdir(path + "/" + str(pt))
        for arq in arquivos:
            curTextImg = cv2.imread(path + "/" + str(pt) + "/" + arq, cv2.IMREAD_GRAYSCALE)
            text_images.append(curTextImg)
            text_classNo.append(pt)

    text_images = np.array(text_images)
    text_classNo = np.array(text_classNo)

    text_images = np.array(list(map(preprocessing, text_images)))
    text_images = text_images.reshape(text_images.shape[0], text_images.shape[1], text_images.shape[2], 1)

    return text_images, text_classNo

# Carregamento e divisão das imagens de texto
X_text, y_text = load_and_process_text(path_text)
X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X_text, y_text, test_size=0.2)
X_text_train, X_text_validation, y_text_train, y_text_validation = train_test_split(
    X_text_train, y_text_train, test_size=0.2
)

# Carregamento e divisão das imagens visuais
count = 0
images = []
classNo = []
pastas = os.listdir(path_visual)
print("Total de Classes:", len(pastas))
noOfClasses = len(pastas)

for pt in range(0, len(pastas)):
    arquivos = os.listdir(path_visual + "/" + str(count))
    for arq in arquivos:
        curImg = cv2.imread(path_visual + "/" + str(count) + "/" + arq)
        images.append(curImg)
        classNo.append(count)

    count += 1

images = np.array(images)
classNo = np.array(classNo)

# Divisão das imagens visuais
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

# Pré-processamento das imagens visuais
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Aumento de imagens
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# One-hot encoding das classes
y_train = tf.keras.utils.to_categorical(y_train, noOfClasses)
y_validation = tf.keras.utils.to_categorical(y_validation, noOfClasses)
y_test = tf.keras.utils.to_categorical(y_test, noOfClasses)

# Criação do modelo
def myModel():
    input_visual = Input(shape=(32, 32, 1), name='input_visual')
    conv_visual = keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(input_visual)
    maxpool_visual = keras.layers.MaxPooling2D(2, 2)(conv_visual)
    flatten_visual = Flatten()(maxpool_visual)

    input_text = Input(shape=(32, 32, 1), name='input_text')
    conv_text = keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(input_text)
    maxpool_text = keras.layers.MaxPooling2D(2, 2)(conv_text)
    flatten_text = Flatten()(maxpool_text)

    concatenated = concatenate([flatten_visual, flatten_text])

    dense1 = Dense(128, activation='relu')(concatenated)
    output = Dense(noOfClasses, activation='softmax')(dense1)

    model = Model(inputs=[input_visual, input_text], outputs=output)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Treinamento do modelo
model = myModel()
print(model.summary())

history = model.fit(
    [X_train, X_text_train], y_train,
    validation_data=([X_validation, X_text_validation], y_validation),
    steps_per_epoch=steps_per_epoch_val, epochs=epochs_val, shuffle=1
)

# Histórico de treinamento
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# Avaliação do modelo
score = model.evaluate([X_test, X_text_test], y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Salvando o modelo
model.save('modelo.h5')
print('Modelo Salvo!')