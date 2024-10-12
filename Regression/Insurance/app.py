import utils
from utils import Model
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

age = int(input('Age: '))
sex = input('sex: ')
bmi = float(input('bmi: '))
children = int(input('children: '))
smoke = input('smoker: ')
region = input('region: ')
numerical_cols = ['age','bmi','children']
data = {'age':[age], 'sex':[sex],'bmi':[bmi],'children':[children],'smoker':[smoke],'region':[region]}
data = pd.DataFrame(data)

print('Following are your details\n',data)

data = utils.add_na_cols(data)

ModelRF = utils.load_object('ModelRF.pck')
data = utils.fill_categorical_cols(data,ModelRF.column_modes)
data = utils.handle_outliers(data,ModelRF.outlier_stats)
data = utils.apply_imputer(data,numerical_cols,ModelRF.Imputer)

data['age'] = utils.round_it_off(data['age'].values)
data['children'] = utils.round_it_off(data['children'].values)

data = utils.apply_labelencoders(data,ModelRF.column_Le)
data = utils.apply_Ohe(data,ModelRF.column_Ohe)
data = utils.apply_scaler(data,numerical_cols,ModelRF.Scaler)

predictions = ModelRF.model.predict(data)
print('The amount to be paid by the you is ', round(int(predictions[0]),2))