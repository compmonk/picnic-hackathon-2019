from keras.models import *
from keras.layers import *
from keras import backend as K

IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150

if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (3, IMAGE_WIDTH, IMAGE_HEIGHT)
else:
    INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)


def CNN():
    model = Sequential()

    model.add(Conv2D(filters=4, kernel_size=2, padding='same',
                     activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=12, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(25, activation='softmax'))

    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
