from shutil import copyfile, copy

import pandas as pd
import numpy as np

from settings import TRAIN_DIR, DATA_DIR, SAMPLE_DIR, SAMPLE_NUMBER, logger


def create_directories(base, n, labels):
    for i in range(1, n + 1):
        for label in labels:
            logger.info("creating class directory for sample: {0}, class: {1}".format(i, label))
            base.child(str(i)).child("train").child(label).mkdir(parents=True)
            base.child(str(i)).child("test").child(label).mkdir(parents=True)


def sample(number_of_samples):
    logger.info("removing old sample directory")
    SAMPLE_DIR.rmtree()

    training_data = pd.read_csv(DATA_DIR.child('train.tsv'), sep='\t')
    training_data['file_path'] = training_data['file'].apply(lambda f: TRAIN_DIR.child(f))

    pd.DataFrame(training_data['label'].unique()).to_csv(DATA_DIR.child("class.csv"), index=False, header=None)

    create_directories(SAMPLE_DIR, number_of_samples, training_data['label'].unique())

    # generate random samples
    for i in range(1, number_of_samples + 1):
        logger.info("generating sample number {0}".format(i))
        idx = np.random.rand(len(training_data)) <= 0.8
        train = training_data[idx]
        train.to_csv(SAMPLE_DIR.child(str(i)).child("train.csv".format(i)), sep=',', index=False)

        test = training_data[~idx]
        test.to_csv(SAMPLE_DIR.child(str(i)).child("test.csv".format(i)), sep=',', index=False)

        # populate random samples
        for index, image in train.iterrows():
            # print(image)
            logger.info("copying {0} to {1}".format(image['file_path'],
                                                    SAMPLE_DIR.child(str(i)).child("train").child(image['label'])))
            # image['file_path'].copy(SAMPLE_DIR.child(str(i)).child("train").child(image['label']))
            copy(image['file_path'], SAMPLE_DIR.child(str(i)).child("train").child(image['label']))

        for index, image in test.iterrows():
            logger.info("copying {0} to {1}".format(image['file_path'],
                                                    SAMPLE_DIR.child(str(i)).child("test").child(image['label'])))
            # image['file_path'].copy("{0}/".format(SAMPLE_DIR.child(str(i)).child("test").child(image['label'])))
            copy(image['file_path'], SAMPLE_DIR.child(str(i)).child("test").child(image['label']))


if __name__ == '__main__':
    sample(number_of_samples=SAMPLE_NUMBER)
