import math
import pandas as pd
import numpy as np

# calculate entropy of a given output column
def entropy(y):
    if isinstance(y, pd.Series):
        p = y.value_counts() / y.shape[0]
        return -1 * sum([(pv * np.log2(pv)) for pv in p])
    else:
        raise TypeError("y must be pandas Series!")


# calculate remainder
def remainder():
    pass


# calculate gain of a given attribute for a given entropy
def gain():
    pass


# choose feature to use as splitting criterion based on gain
def split_criteria():
    pass
