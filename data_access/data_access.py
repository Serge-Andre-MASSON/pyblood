from io import BytesIO
from PIL import Image
import pickle

import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage

import os
from dotenv import load_dotenv

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
    if DATA_ACCESS == 'local':
        print(DATA_ACCESS)
        return Image.open(img_path)
    else:
        print(DATA_ACCESS)
        img_as_bytes = BytesIO(BUCKET.blob(img_path).download_as_bytes())
        return Image.open(img_as_bytes)


def get_dataset_infos():
    """Return a DataFrame containing path , width, height and cell's type for each image."""

    with open('data_access/PBC_infos.PICKLE', 'rb') as f:
        PBC_infos_df = pickle.load(f)
    return PBC_infos_df


def load_pickle_data(name):
    data_path = f'data/PBC_pickles/{name}.PICKLE'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_pickle_selector(name):
    data_path = f'data/feature_selection/{name}.PICKLE'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data
