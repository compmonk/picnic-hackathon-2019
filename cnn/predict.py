import pandas as pd
from keras_preprocessing.image import *

from cnn.CNN import CNN, INPUT_SHAPE
from settings import SAMPLE_NUMBER, SAMPLE_DIR, logger, TEST_DIR, DATA_DIR


def predict(base):
    model = CNN()
    logger.info("loading weights from {0}".format(base.child("model.h5")))
    model.load_weights(base.child("model.h5"))

    labels = pd.read_csv(DATA_DIR.child("class.csv"))

    output_df = pd.DataFrame(columns=['file', 'label'])
    for image in TEST_DIR.listdir():
        img = load_img(image, target_size=INPUT_SHAPE)
        label = model.predict_classes(np.expand_dims(img_to_array(img) / 255, axis=0))[0]
        output_df = output_df.append({'file': image.name, 'label': labels['label'][label]}, ignore_index=True)
        logger.info("{0} is classified as {1}".format(image.name, labels['label'][label]))

    logger.info("saving classifications at {0}".format(base.child("output.tsv")))
    output_df.to_csv(base.child("output.tsv"), sep='\t', index=False)


if __name__ == '__main__':
    for i in range(1, SAMPLE_NUMBER):
        predict(SAMPLE_DIR.child(str(i)))
