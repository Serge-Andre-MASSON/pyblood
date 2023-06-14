import tensorflow as tf
from io import BytesIO
from random import randint
from PIL import Image
import pickle

import h5py
import streamlit as st
from google.cloud import storage
import pandas as pd

import os
from dotenv import load_dotenv

from data_access.data_paths import get_pbc_dataset_infos_paths

from joblib import load

BUCKET = ''
load_dotenv()
DATA_ACCESS = os.getenv("DATA_ACCESS")


if DATA_ACCESS != 'local':
    client = storage.Client()

    BUCKET_NAME = "pyblood_bucket"
    BUCKET = client.bucket(BUCKET_NAME)
    DATA_ACCESS = 'google clood'


def get_image(path):
    """Return the image or figure located at img_path."""
    if DATA_ACCESS != 'local':
        path = BytesIO(BUCKET.blob(path).download_as_bytes())

    with Image.open(path) as f:
        img = f.copy()

    return img


def get_random_image():
    """Return a random image picked from the original dataset, together with is cell's type."""
    dataset_infos_path = get_pbc_dataset_infos_paths('both')
    dataset_infos = load_pickle(dataset_infos_path)

    random_id = randint(0, len(dataset_infos) - 1)
    img_path, cell_type = dataset_infos.iloc[random_id]
    return get_image(img_path), cell_type


def load_pickle(path):
    """Return unpickled data located at path."""
    if DATA_ACCESS != 'local':
        path = BytesIO(
            BUCKET.blob(path)
                  .download_as_bytes())

        p = pd.read_pickle(path)
    else:
        with open(path, 'rb') as f:
            p = pd.read_pickle(path)
    return p


@st.cache(allow_output_mutation=True)
def load_dl_model(path):
    """Load the model located at the specified path."""
    if DATA_ACCESS != 'local':
        path = BytesIO(BUCKET.blob(path).download_as_bytes())
        path = h5py.File(path)

    model = tf.keras.models.load_model(path)
    return model


@st.cache(allow_output_mutation=True)
def load_ml_model(path):
    """Load the model located at the specified path."""
    if DATA_ACCESS != 'local':
        path = BytesIO(BUCKET.blob(path).download_as_bytes())

    model = load(path)
    return model
