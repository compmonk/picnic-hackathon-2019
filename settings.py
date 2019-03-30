import logging as logger
import sys

from unipath import Path

BASE_DIR = Path(__file__).ancestor(1)
DATA_DIR = BASE_DIR.child("data")
TRAIN_DIR = DATA_DIR.child("train")
TEST_DIR = DATA_DIR.child("test")
SAMPLE_DIR = DATA_DIR.child("sample")

SAMPLE_NUMBER = 5

logger.basicConfig(stream=sys.stderr, level=logger.INFO)
