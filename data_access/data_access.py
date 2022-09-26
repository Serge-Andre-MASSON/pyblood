from io import BytesIO
from random import randint
from PIL import Image
import pickle

import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage

import os
from dotenv import load_dotenv

from data_access.data_paths import get_pbc_dataset_infos_paths

BUCKET = ''
load_dotenv()
DATA_ACCESS = os.getenv("DATA_ACCESS")


if DATA_ACCESS != 'local':
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"])
    client = storage.Client(credentials=credentials)

    BUCKET_NAME = "pyblood_bucket"
    BUCKET = client.bucket(BUCKET_NAME)
    DATA_ACCESS = 'google clood'


def get_image(img_path):
    """Return the image located at img_path."""
    if DATA_ACCESS != 'local':
        img_path = BytesIO(BUCKET.blob(img_path).download_as_bytes())

    with Image.open(img_path) as f:
        img = f.copy()

    return img


def get_random_image():
    dataset_infos_path = get_pbc_dataset_infos_paths('both')
    dataset_infos = load_pickle(dataset_infos_path)

    random_id = randint(0, len(dataset_infos) - 1)
    img_path, cell_type = dataset_infos.iloc[random_id]
    return get_image(img_path), cell_type


def load_pickle(path):
    """Return unpickled data located at path."""
    if DATA_ACCESS != 'local':
        print(path)
        path = BytesIO(
            BUCKET.blob(path)
                  .download_as_bytes())
        p = pickle.load(path)
    else:
        with open(path, 'rb') as f:
            p = pickle.load(f)
    return p
