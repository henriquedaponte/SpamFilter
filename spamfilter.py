import numpy as np
import pandas as pd



def loadData(filename):

    data = pd.read_csv(filename, delimiter=',')

    return 0 # STUB