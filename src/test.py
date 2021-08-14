import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sklearn.metrics as metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # criando argumentos para linha de comando
# ap = argparse.ArgumentParser()
# ap.add_argument("--mode",help="train/display")
# mode = ap.parse_args().mode

# plots accuracy and loss curves
# def plot_model_history(model_history):
    # """
    # Plot Accuracy and Loss curves given the model_history
    # """
    # fig, axs = plt.subplots(1,2,figsize=(15,5))
    # # summarize history for accuracy
    # axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    # axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    # axs[0].set_title('Model Accuracy')
    # axs[0].set_ylabel('Accuracy')
    # axs[0].set_xlabel('Epoch')
    # # axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    # axs[0].set_xticks(np.arange(num_epoch))
    # axs[0].legend(['train', 'val'], loc='best')
    # # summarize history for loss
    # axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    # axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    # axs[1].set_title('Model Loss')
    # axs[1].set_ylabel('Loss')
    # axs[1].set_xlabel('Epoch')
    # # axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    # axs[0].set_xticks(np.arange(num_epoch))
    # axs[1].legend(['train', 'val'], loc='best')
    # fig.savefig('plot.png')
    # plt.show()

# path para os datasets
train_dir = 'data/train'
val_dir = 'data/test'
# define tamanho para os datasets de treino e validação
num_train = 28709
num_val = 7178
batch_size = 64#tamanho dos lotes a cada iteração
num_epoch = 50 #número de épocas

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

#gera os lotes de imagens para treinamento em escalas de cinza
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

#gera os lotes de imagens para validação em escalas de cinza
test_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')




# recriando o modelo...
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# model.load_weights('model_50_epochs.h5')
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
model_info = model.fit_generator(
    train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=num_epoch,
    validation_data=test_generator,
    validation_steps=num_val // batch_size)
model.save_weights('model.h5')
# plot_model_history(model_info)



test_steps_per_Epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
predictions = model.predict_generator(test_generator, steps=test_steps_per_Epoch)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
accuracy_score = metrics.accuracy_score(true_classes, predicted_classes)
print(report)
print(accuracy_score)

