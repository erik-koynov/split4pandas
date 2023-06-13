import pandas as pd
import numpy as np
from  split4pandas import SplitGenerator, ModelSelectionMode


df = pd.DataFrame({"a": list(range(10)),"b": np.random.random(10)>0.5})
print(df.b.value_counts())
split_generator = SplitGenerator(df, stratify='b', mode=ModelSelectionMode.KFOLD)

for training_set, test_set in split_generator(test_size=0.2):
    print("training set: ", training_set)
    print("test_set: ", test_set)