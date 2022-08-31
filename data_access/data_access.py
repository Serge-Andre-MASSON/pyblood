from io import BytesIO
from PIL import Image
import pickle

import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

bucket_name = "pyblood_bucket"

bucket = client.bucket(bucket_name)


def get_image(img_path):
    """Return the image located at img_path on the a google cloud storage."""

    img_as_bytes = bucket.blob(img_path).download_as_bytes()

    return Image.open(BytesIO(img_as_bytes))


def get_dataset_infos():
    """Return a DataFrame containing path , width, height and cell's type for each image."""

    with open('data_access/PBC_infos.PICKLE', 'rb') as f:
        PBC_infos_df = pickle.load(f)
    return PBC_infos_df
