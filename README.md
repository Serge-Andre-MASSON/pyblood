# Pyblood
datascientest project

app : https://serge-andre-masson-pyblood-app-zgh56j.streamlitapp.com/

# Usage

### Create a conda env with python 3.9 and activate it:

$ conda create -n <env_name> python=3.9

$ conda activate <env_name>

### Use pip to install required libraries within <env_name>:

$ pip install -r requirements.txt

### Local data access

At the root of the project, create a dotenv file named .env and write in it the following line :

DATA_ACCESS=local

To download* the data for the project on your local workstation :

$ ipython

from data_access.data_update import download_data

download_data()


*For this procedure to work you'll need to have acces to the .json containing the google cloud storage credentials.

