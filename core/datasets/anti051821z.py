import typing
from core.datasets import CSVDataset


class ANTI051821Z(CSVDataset):
    """

    ## Parameters:
      - include_outlier: bool - include rows that have column `Is Outlier` == Yes
      - include_MSI011320: bool - include rows that are already present on MSI011320 dataset (should always be fault)
      - min_evidence: int - how many evidence is at least needed to be include
    """

    def __init__(self, include_outlier: bool, test_split_method: typing.Literal['has_binding'] = 'has_binding', include_MSI011320: bool = False, min_evidence: int = 1) -> None:
        super().__init__()
        # we use split test set wether or not it has binding data (from experiment)
        assert test_split_method == 'has_binding', f"Not supported test split method {test_split_method}"
