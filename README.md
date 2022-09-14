# Pyblood
datascientest project

app : https://serge-andre-masson-pyblood-app-zgh56j.streamlitapp.com/

# Usage

### Create a conda env  and activate it:

$ conda create -n <env_name>

$ conda activate <env_name>

### Install pytorch with conda following instructions at :

https://pytorch.org/get-started/locally/

### Use pip to install required libraries within <env_name>:

$ pip install -r requirements.txt

### Local data access

At the root of the project, create a dotenv file named .env and write in it the following line :

DATA_ACCESS=local

To use local data access, you need to have a directory named data in the root of the project, containing PBC_dataset_normal_DIB ( link: https://drive.google.com/file/d/1gKWvyIYfi0JXnLrLA3TM6W1WKiTRjmvM/view?usp=sharing) 
