from flask import Flask, request, jsonify
import pandas as pd
import utils
from utils import Model

app = Flask(__name__)

# Load the pre-trained model (assumed to be in a pickle file)
ModelRF = utils.load_object('ModelRF.pck')

@app.route('/')
def home():
    return '''
    <h1>Insurance Prediction API</h1>
    <p>Use the /predict endpoint to get predictions via a POST request.</p>
    '''


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        input_data = request.json
        
        # Extract input variables
        age = input_data.get('age')
        sex = input_data.get('sex')
        bmi = input_data.get('bmi')
        children = input_data.get('children')
        smoke = input_data.get('smoker')
        region = input_data.get('region')

        # Create a DataFrame from input
        data = {'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children], 'smoker': [smoke], 'region': [region]}
        data = pd.DataFrame(data)

        # Preprocessing steps using utils
        numerical_cols = ['age', 'bmi', 'children']
        data = utils.add_na_cols(data)
        data = utils.fill_categorical_cols(data, ModelRF.column_modes)
        data = utils.handle_outliers(data, ModelRF.outlier_stats)
        data = utils.apply_imputer(data, numerical_cols, ModelRF.Imputer)

        # Round off age and children columns
        data['age'] = utils.round_it_off(data['age'].values)
        data['children'] = utils.round_it_off(data['children'].values)

        # Apply encoders and scaling
        data = utils.apply_labelencoders(data, ModelRF.column_Le)
        data = utils.apply_Ohe(data, ModelRF.column_Ohe)
        data = utils.apply_scaler(data, numerical_cols, ModelRF.Scaler)

        # Make the prediction
        predictions = ModelRF.model.predict(data)

        # Send the result back as a response
        return jsonify({'message': 'Prediction successful', 'amount': round(predictions[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
