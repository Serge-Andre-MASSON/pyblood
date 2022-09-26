from pathlib import Path


TRANSFORMER_ROOT = Path("data/transformers")
PBC_PICKLES_ROOT = Path("data/PBC_pickles")


def get_transformer_path(size, *args):
    transformers = '_'.join(args)
    return TRANSFORMER_ROOT/f"{transformers}_{size}.PICKLE"


def get_pbc_dataset_infos_paths(name: str):
    """Return the path to dataset infos. 
    name can be 'paths' for images's paths, 'targets for targets's paths or 'both'."""
    if name == 'paths':
        return PBC_PICKLES_ROOT / "paths.PICKLE"
    elif name == 'targets':
        return PBC_PICKLES_ROOT / "targets.PICKLE"
    elif name == 'both':
        return PBC_PICKLES_ROOT / "dataset_infos.PICKLE"


if __name__ == "__main__":
    pass
