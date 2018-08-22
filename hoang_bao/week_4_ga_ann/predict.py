import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from GA import GANN
import operator

class NN:
    file_name ='./data_resource_usage_5Minutes_6176858948.csv'
    df = pd.read_csv(file_name,header=None)
    def __init__(self):
        self.cpu = list(df[3])
        self.mem = list(df[4])
        #xac dinh train size va test size
        train_size = int(len(cpu)*0.8)
        test_size = len(cpu) - train_size
    def fitness(X)