import os
import random

import numpy as np


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    return s
