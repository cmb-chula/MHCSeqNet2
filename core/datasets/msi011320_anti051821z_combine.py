from __future__ import annotations

from core.datasets.csv_datasets import CSVDataset
import os
import typing


class MSI011320_ANTI051821Z_COMBINE(CSVDataset):
    """
    A dataset from base file `HLA_classI_MS_dataset_011320.tsv`
    ## Parameters:
      - kfold: int - tell which fold to load `train_{k}.csv` and `test_{k}.csv`
    """
    home_dir = 'MSI011320_ANTI051821Z_COMBINE'

    @classmethod
    def get_csv_path(cls, kfold: int, phase: typing.Literal['train', 'eval', 'test'], root_dir: typing.Optional[str] = None):
        return os.path.join(root_dir or cls.root_dir, cls.home_dir, f"HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-{kfold}_{phase}.csv")
