import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
DATA_ACCESS = os.getenv("DATA_ACCESS")


TRANSFORMER_ROOT = Path("data/transformers")
PBC_PICKLES_ROOT = Path("data/PBC_pickles")


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
