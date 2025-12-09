# détection classif/regression
import os 
import pandas as pd


def detect_problem_type(df_solution):
    df=df_solution
    df=df.head(1000)
    if len(df.columns)==1:
        unique_values=df[df.columns[0]].nunique()
        
        if (unique_values)<20:
            return 1, False #classification one label
        else:
            return 0, False #regression
    else:
        
        if len(df[df.sum(axis=1)==1])==len(df):
            return 1, True #classification multi label (one hot encoded)
        else: 
            return 2 ,False#classification multi class