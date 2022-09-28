import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
DATA_ACCESS = os.getenv("DATA_ACCESS")


TRANSFORMER_ROOT = Path("data/transformers")
PBC_PICKLES_ROOT = Path("data/PBC_pickles")
IMAGES_ROOT = Path("data/images")
ML_MODELS_ROOT = Path("data/ml_models")

def get_image_path(name: str):
    if name == 'performance vs size':
        path = IMAGES_ROOT / "performance_vs_size.png"
    return get_correct_path(path)

def get_ml_model_path(name: str):
    if name == 'svc_30':
        path = ML_MODELS_ROOT / "svc_30.joblib"
    if name == 'svc_70':
        path = ML_MODELS_ROOT / "svc_70.joblib"
    if name == 'svc_100':
        path = ML_MODELS_ROOT / "svc_100.joblib"
    if name == 'svc_200':
        path = ML_MODELS_ROOT / "svc_200.joblib"
    if name == 'rfc_30':
        path = ML_MODELS_ROOT / "rfc_30.joblib"
    if name == 'rfc_70':
        path = ML_MODELS_ROOT / "rfc_70.joblib"
    if name == 'rfc_100':
        path = ML_MODELS_ROOT / "rfc_100.joblib"
    if name == 'rfc_200':
        path = ML_MODELS_ROOT / "rfc_200.joblib"
    return get_correct_path(path)

def get_correct_path(path):
    if DATA_ACCESS != 'local':
        return str(path)
    else:
        return path


def get_transformer_path(size, *args):
    transformers = '_'.join(args)
    path = TRANSFORMER_ROOT/f"{transformers}_{size}.PICKLE"
    return get_correct_path(path)


def get_pbc_dataset_infos_paths(name: str):
    """Return the path to dataset infos. 
    name can be 'paths' for images's paths, 'targets for targets's paths or 'both'."""
    if name == 'paths':
        path = PBC_PICKLES_ROOT / "paths.PICKLE"
    elif name == 'targets':
        path = PBC_PICKLES_ROOT / "targets.PICKLE"
    elif name == 'both':
        path = PBC_PICKLES_ROOT / "dataset_infos.PICKLE"
    return get_correct_path(path)


if __name__ == "__main__":
    pass
