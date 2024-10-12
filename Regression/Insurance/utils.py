import pickle
import numpy as np
import pandas as pd

class Model:
    def __init__(self, column_modes, outlier_stats, Imputer, column_Le, column_Ohe, Scaler, model):
        self.column_modes = column_modes
        self.outlier_stats = outlier_stats
        self.Imputer = Imputer
        self.column_Le = column_Le
        self.column_Ohe = column_Ohe
        self.Scaler = Scaler
        self.model = model

def load_object(file_path):
    with open(file_path, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object

# adding nan columns
def add_na_cols(df):
    for col in df.columns:
        df[col+"_na"] = df[col].isnull().astype(int)
    return(df)

# fill categorical cols
def fill_categorical_cols(df,column_modes):
    for col, value in column_modes.items():
        df[col].fillna(value=value, inplace=True)
    return(df)

# handling outliers
def handle_outliers(df,outlier_stats):
    for col, outlier_stat in outlier_stats.items():
        idx = df.index[(df[col]>outlier_stat['upper_limit']) | (df[col]<outlier_stat['lower_limit'])]
        df.loc[idx,col] = np.nan
    return df

# Apply KNN imputer
def apply_imputer(df,numerical_cols,Imputer):
    df.loc[:,numerical_cols] = Imputer.transform(df.loc[:,numerical_cols])
    return(df)

# Round off age and children
def round_it_off(values):
    return(np.floor(values+0.5).astype(int))

# Apply Label Encoder
def apply_labelencoders(df,columns_Le):
    for col, Le in columns_Le.items():
        df[col] = Le.transform(df[col])
    return(df)

# Apply OHE
def apply_Ohe(df,columns_ohe):
    for col, ohe in columns_ohe.items():
        ohe_output = ohe.transform(df[[col]]) # returns 2d numpy array
        col_names = [col + "_" + str(val) for val in ohe.categories_[0]]
        ohe_output = pd.DataFrame(ohe_output[:,:-1], columns=col_names[:-1])
        df.drop([col], axis=1, inplace=True)
        df = pd.concat([df, ohe_output], axis=1)
    return df

# apply Scaler
def apply_scaler(df,cols_to_scale,scaler):
    df.loc[:,cols_to_scale] = scaler.transform(df[cols_to_scale])
    return(df)