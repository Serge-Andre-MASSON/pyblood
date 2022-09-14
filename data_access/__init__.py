import os
from dotenv import load_dotenv


load_dotenv()
DATA_ACCESS = os.getenv("DATA_ACCESS")

if DATA_ACCESS == "local":
    from data_access.local_data_access import get_image, get_dataset_infos
else:
    from data_access.data_access import get_image, get_dataset_infos
    