from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import pandas as pd
from typing import Union
from .model_selection_mode import ModelSelectionMode


class SplitGenerator:
    def __init__(self, dataset: pd.DataFrame,
                 mode: ModelSelectionMode = ModelSelectionMode.TRAINTEST,
                 stratify: Union[str, pd.Series] = None):
        self.input_ds = dataset.copy().reset_index(inplace=False, drop=True)
        self.mode = mode

        if isinstance(stratify, str):
            stratify = self.input_ds[stratify]

        self.stratify = stratify

    def __call__(self, test_size=0.2):
        for train, test in SplitGenerator.split_generator(self.input_ds,
                                                          test_size=test_size,
                                                          stratify=self.stratify,
                                                          mode=self.mode):

            yield self.input_ds.iloc[train], self.input_ds.iloc[test]


    @staticmethod
    def split_generator(dataset,
                        test_size=0.2,
                        stratify=None,
                        mode: ModelSelectionMode = ModelSelectionMode.TRAINTEST):
        dataset_index = list(dataset.index)
        if mode == ModelSelectionMode.KFOLD:

            test_size = int(1. / test_size)
            if stratify is not None:
                kfold = StratifiedKFold(n_splits=test_size, shuffle=True, random_state=42)
                splits = kfold.split(dataset_index, stratify)

            else:
                kfold = KFold(n_splits=test_size, shuffle=True, random_state=42)
                splits = kfold.split(dataset_index)

        elif mode == ModelSelectionMode.TRAINTEST:

            splits = [train_test_split(dataset_index,
                                       shuffle=True,
                                       random_state=42,
                                       stratify=stratify)]

        else:
            raise Exception("mode not allowed.")

        for train, test in splits:
            yield train, test
