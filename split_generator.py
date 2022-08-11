from sklearn.model_selection import train_test_split, KFold
import pandas as pd
class ModelSelectionMode:
    KFOLD = 1
    TRAINTEST = 2

class SplitGenerator:
    def __init__(self, dataset: pd.DataFrame, mode: int = ModelSelectionMode.TRAINTEST):
        self.input_ds = dataset
        self.mode = mode

    def __call__(self, test_size=0.2):
        for train, test in SplitGenerator.split_generator(self.input_ds, test_size=test_size, mode=self.mode):
            yield self.input_ds.iloc[train], self.input_ds.iloc[test]


    @staticmethod
    def split_generator(dataset, test_size=0.2, mode: int = ModelSelectionMode.TRAINTEST):
        dataset_index = list(dataset.index)
        if mode == ModelSelectionMode.KFOLD:
            test_size = int(1. / test_size)
            kfold = KFold(n_splits=test_size, shuffle=True, random_state=42)
            splits = kfold.split(dataset_index)

        elif mode == ModelSelectionMode.TRAINTEST:
            splits = [train_test_split(dataset_index, shuffle=True, random_state=42)]

        else:
            raise Exception("mode not allowed.")

        for train, test in splits:
            yield train, test

if __name__=="__main__":
    df = pd.DataFrame({"a": list(range(10)),"b": list(range(10))})
    split_generator = SplitGenerator(df)
    for training_set, test_set in split_generator(test_size = 0.2):
        print("training set: ", training_set)
        print("test_set: ", test_set)