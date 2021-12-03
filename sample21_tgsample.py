import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple,NamedTuple
from collections import OrderedDict
from functools import partial
import tarfile
from urllib import request
# def get_download_path() -> Path:
#     return Path(os.environ.get("MXNET_HOME", str(Path.home() / ".mxnet" / "gluon-ts")))
# default_dataset_path = get_download_path() / "datasets"
default_dataset_path ='.'
print(default_dataset_path)

class GPCopulaDataset(NamedTuple):
    name: str
    url: str
    num_series: int
    prediction_length: int
    freq: str
    rolling_evaluations: int
    max_target_dim: Optional[int] = None

root = "https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/"
datasets_info = {
    "electricity_nips": GPCopulaDataset(
        name="electricity_nips",
        url=root + "electricity_nips.tar.gz",
        # original dataset can be found at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#
        num_series=370,
        prediction_length=24,
        freq="H",
        rolling_evaluations=7,
        max_target_dim=None,
    ),}

def download_dataset(dataset_path: Path, ds_info: GPCopulaDataset):
    request.urlretrieve(ds_info.url, dataset_path / f"{ds_info.name}.tar.gz")

    with tarfile.open(dataset_path / f"{ds_info.name}.tar.gz") as tar:
        tar.extractall(path=dataset_path)

def generate_gp_copula_dataset(
    dataset_path: Path,
    dataset_name: str,
    prediction_length: Optional[int] = None,
):
    ds_info = datasets_info[dataset_name]
    os.makedirs(dataset_path, exist_ok=True)

    download_dataset(dataset_path.parent, ds_info)
    save_metadata(dataset_path, ds_info, prediction_length)
    save_dataset(dataset_path / "train", ds_info)
    save_dataset(dataset_path / "test", ds_info)
    clean_up_dataset(dataset_path, ds_info)

dataset_recipes = OrderedDict(
    {"electricity_nips": partial(generate_gp_copula_dataset, dataset_name="electricity_nips"),})


def materialize_dataset(dataset_name: str, path: Path = default_dataset_path,
    regenerate: bool = False,prediction_length: Optional[int] = None,) -> Path:
    assert dataset_name in dataset_recipes.keys(), (
        f"{dataset_name} is not present, please choose one from "
        f"{dataset_recipes.keys()}."
    )

    path.mkdir(parents=True, exist_ok=True)
    dataset_path = path / dataset_name

    dataset_recipe = dataset_recipes[dataset_name]
    if not dataset_path.exists() or regenerate:
        logging.info(f"downloading and processing {dataset_name}")
        if dataset_path.exists():
            # If regenerating, we need to remove the directory contents
            shutil.rmtree(dataset_path)
            dataset_path.mkdir()
        # Optionally pass prediction length to not override any non-None
        # defaults (e.g. for M4)
        kwargs: Dict[str, Any] = {"dataset_path": dataset_path}
        if prediction_length is not None:
            kwargs["prediction_length"] = prediction_length
        dataset_recipe(**kwargs)
    else:
        logging.info(
            f"using dataset already processed in path {dataset_path}."
        )
    return dataset_path

def get_dataset(dataset_name: str,path: Path = default_dataset_path,regenerate: bool = False,
    prediction_length: Optional[int] = None,):

    dataset_path = materialize_dataset(
        dataset_name, path, regenerate, prediction_length
    )

    return load_datasets(metadata=dataset_path,train=dataset_path / "train",test=dataset_path / "test")
