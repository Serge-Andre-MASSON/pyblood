import os
from pathlib import Path
from dotenv import load_dotenv

"""This module is supposed to build all the paths that the application needs. 
The purpose is to avoid to use hard coded paths.

Practicaly, you may use hard coded path at first and then, create a function that build them here if possible/necessary."""


BUCKET = ''
load_dotenv()
DATA_ACCESS = os.getenv("DATA_ACCESS")

TRANSFORMER_ROOT = Path("data/transformers")
PBC_PICKLES_ROOT = Path("data/PBC_pickles")
ML_MODELS_ROOT = Path("data/ml_models")
DL_MODELS_ROOT = Path("data/dl_models")
FIGURES_ROOT = Path("data/figures")


def get_correct_path(path):
    if DATA_ACCESS != "local":
        return str(path).replace('\\', '/')
    return path


def get_figure_path(figure_name, extension='jpg'):
    path = FIGURES_ROOT / f"{figure_name}.{extension}"
    return get_correct_path(path)


def get_transformer_path(size, *args):
    """Return the transformer associated with the size. Possible argument after the size are 
    'sp', 'pca', or 'crop'.

    For example, if you need the transformer that combine SelectPercentile 
    and pca for 50 sized images you may write:

    transformer_path =  get_transformer_path(50, 'sp', 'pca').

    Why this path you may then call the load_pickle (from data_access.data_access) function to access the desired content."""
    transformers = '_'.join(args)
    path = TRANSFORMER_ROOT/f"{transformers}_{size}.PICKLE"
    return get_correct_path(path)


def get_pbc_dataset_infos_paths(name: str):
    """Return the path to dataset infos. 
    name can be 'paths' for images's paths, 'targets' for targets's paths or 'both' to get a dataframe that combine both."""
    if name == 'paths':
        path = PBC_PICKLES_ROOT / "paths.PICKLE"
    elif name == 'targets':
        path = PBC_PICKLES_ROOT / "targets.PICKLE"
    elif name == 'both':
        path = PBC_PICKLES_ROOT / "dataset_infos.PICKLE"
    return get_correct_path(path)


def get_dl_model_path(model_name):
    path = DL_MODELS_ROOT / f"{model_name}.h5"
    return get_correct_path(path)


def get_dl_mismatch_path(model_name):
    path = DL_MODELS_ROOT / f"{model_name}_mismatch.PICKLE"
    return get_correct_path(path)

def get_ml_mismatch_path(model_name):
    path = ML_MODELS_ROOT / f"{model_name}_mismatch.PICKLE"
    return get_correct_path(path)
