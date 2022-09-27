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
    """Upload all files passed in argument. Local path will match remote path."""
    for file_path in tqdm(file_paths):
        blob = BUCKET.blob(file_path)
        blob.upload_from_filename(file_path)


def download_files(*file_paths):
    """Download all files passed in argument. Remote path will match local path."""
    for file_path in tqdm(file_paths):
        blob = BUCKET.blob(file_path)
        blob.download_to_filename(file_path)


def upload_data():
    """Browse local data and upload in the cloud the missing elements."""
    for root, _, files in os.walk("data/"):
        if files and "PBC_dataset_normal_DIB" not in root:
            for file in files:
                file_path = root + "/" + file
                blob = BUCKET.blob(file_path)
                if not blob.exists():
                    print("Uploading :", file_path)
                    blob.upload_from_filename(file_path, timeout=4000)


def download_data():
    """Browse remote data and download locally the missing elements."""
    blobs = client.list_blobs(BUCKET)
    for blob in blobs:
        blob_name = blob.name
        if not os.path.exists(blob_name):
            if blob_name[-1] == '/':
                os.mkdir(blob_name)
            else:
                try:
                    os.mkdir(Path(*blob_name.split('/')[:-1]))
                except FileExistsError:
                    pass

            try:
                BUCKET.blob(blob_name).download_to_filename(blob_name)
            except IsADirectoryError:
                print(
                    f"This blob ({blob}) is a directory, go to the next one...")


if __name__ == "__main__":
    download_data()
