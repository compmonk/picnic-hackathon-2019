from keras.preprocessing.image import ImageDataGenerator

from settings import SAMPLE_DIR, SAMPLE_NUMBER, logger
from cnn.CNN import CNN, IMAGE_WIDTH, IMAGE_HEIGHT


def generate_model(base, train_n, test_n):
    epochs = 50
    batch_size = 16

    model = CNN()

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        base.child("train"),
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        base.child("test"),
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=test_n // batch_size)

    model_path = base.child("model.h5")
    model.save_weights(model_path)

    if model_path.exists():
        logger.info("model saved to {}".format(model_path))
    else:
        logger.warning("unable able to save model")


if __name__ == '__main__':
    for i in range(1, SAMPLE_NUMBER):
        generate_model(SAMPLE_DIR.child(str(i)), 5806, 1492)
