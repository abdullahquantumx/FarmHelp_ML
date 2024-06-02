from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS class
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load the first trained model
crop_yield_model = load_model('model.h5')

# Load the second trained model
acres_prediction_model = load_model('crop_yield_tensorflow_model.h5')

# Define a route for crop yield prediction
# http://localhost:5000/predict?inputData=7,8,4,5,4,5,7
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Retrieve input data from query parameters
        input_data_str = request.args.get('inputData')

        if not input_data_str:
            return jsonify({'error': 'Missing inputData parameter'}), 400

        # Convert input data to a list of floats
        input_data = [float(value) for value in input_data_str.split(',')]

        # Convert input data to a NumPy array
        input_array = np.array([input_data])

        # Make a prediction using the loaded model
        prediction = crop_yield_model.predict(input_array)

        # Get the index of the maximum probability
        predicted_class_index = np.argmax(prediction)

        # Define class labels
        class_labels = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
       'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
       'pigeonpeas', 'pomegranate', 'rice','watermelon']  # Replace with your class labels

        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]

        # Send the prediction result as a JSON response
        return jsonify({'yield_prediction': predicted_class})
    except Exception as e:
        print('Error during prediction:', e)
        return jsonify({'error': 'Internal Server Error'}), 500
    
# Define a route for acres prediction
# http://localhost:5000/predict2?inputData=1230,80,28,80,24,20
@app.route('/predict2', methods=['GET'])
def predict2():
    try:
        # Retrieve input data from query parameters
        input_data_str = request.args.get('inputData')

        if not input_data_str:
            return jsonify({'error': 'Missing inputData parameter'}), 400

        # Convert input data to a list of integers
        input_data = [int(value) for value in input_data_str.split(',')]

        # Check if the number of features matches the model's expectations
        if len(input_data) != 6:
            return jsonify({'error': 'Invalid number of features. Expected 6 features.'}), 400

        # Convert input data to a NumPy array and reshape
        input_array = np.array([input_data]).astype(np.float32)

        # Make a prediction using the loaded model
        prediction = acres_prediction_model.predict(input_array)

        # Convert prediction to integer
        prediction_int = int(prediction)

        # Send the prediction result as a JSON response
        return jsonify({'Acres_prediction': prediction_int})
    except Exception as e:
        print('Error during prediction:', e)
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
