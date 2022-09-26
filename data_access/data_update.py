from pathlib import Path
import math
from tqdm import tqdm
import os
from google.cloud import storage
from google.oauth2 import service_account


credentials = service_account.Credentials.from_service_account_file(
    ".streamlit/pyblood-16bca61a00f2.json")
client = storage.Client(credentials=credentials)

BUCKET_NAME = "pyblood_bucket"
BUCKET = client.bucket(BUCKET_NAME)
DATA_ACCESS = 'google clood'


def upload_files(*file_paths):
    for file_path in tqdm(file_paths):
        blob = BUCKET.blob(file_path)
        blob.upload_from_filename(file_path)


def download_files(*file_paths):
    for file_path in tqdm(file_paths):
        blob = BUCKET.blob(file_path)
        blob.download_to_filename(file_path)


ROOT_URL = "https://storage.googleapis.com/pyblood_bucket/"


def upload_data():
    for root, _, files in os.walk("data/"):
        if files and "PBC_dataset_normal_DIB" not in root:
            for file in files:
                file_path = root + "/" + file
                blob = BUCKET.blob(file_path)
                if not blob.exists():
                    print("Uploading :", file_path)
                    blob.upload_from_filename(file_path, timeout=4000)


def download_data():
    blobs = client.list_blobs(BUCKET)
    for blob in blobs:
        blob_name = blob.name

        if not os.path.exists(blob_name):
            try:
                BUCKET.blob(blob_name).download_to_filename(blob_name)
            except IsADirectoryError:
                os.mkdir(blob_name)
                print(
                    f"This blob ({blob}) is a directory, go to the next one...")
