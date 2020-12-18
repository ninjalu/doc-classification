import numpy as np
from typing import List, Union, Tuple

def f1_score(pred, labels):
    average_f1 = []
    precision = sum(pred=labels)/len(labels)
    recall = 