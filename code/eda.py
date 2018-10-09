import pandas as pd
import numpy as np

from sklearn import preprocessing
from code.util.base_util import get_logger
from code.util.base_util import pickle_load
from code.util.base_util import pickle_dump
from code.util.base_util import timer
import os
import pickle


def normalize_process(df, conti_list):
    for col in conti_list:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
