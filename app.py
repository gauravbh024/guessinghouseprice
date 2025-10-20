from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import gc  # Garbage collector for memory management

app = Flask(__name__)

# Load your ML model and pipeline
MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def load_model_and_pipeline():
    """Load the trained model and preprocessing pipeline from disk"""
    try:
        model = joblib.load(MODEL_FILE)
        pipeline = joblib.load(PIPELINE_FILE)
        print("Model and pipeline loaded successfully!")
        # Force garbage collection to free memory
        gc.collect()
        return model, pipeline
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'model.pkl' and 'pipeline.pkl' are in the same directory as app.py")
        return None, None

model, pipeline = load_model_and_pipeline()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or pipeline is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first using project.py'
            }), 500
        
        # Get data from the form
        data = request.get_json()
        
        # Create a DataFrame with the input features
        # Features must match the order expected by your model
        input_df = pd.DataFrame([{
            'longitude': float(data.get('longitude', 0)),
            'latitude': float(data.get('latitude', 0)),
            'housing_median_age': float(data.get('housing_median_age', 0)),
            'total_rooms': float(data.get('total_rooms', 0)),
            'total_bedrooms': float(data.get('total_bedrooms', 0)),
            'population': float(data.get('population', 0)),
            'households': float(data.get('households', 0)),
            'median_income': float(data.get('median_income', 0)),
            'ocean_proximity': data.get('ocean_proximity', 'INLAND')
        }])
        
        # Transform input using the pipeline
        transformed_input = pipeline.transform(input_df)
        
        # Make prediction
        prediction = model.predict(transformed_input)
        
        return jsonify({
            'success': True,
            'prediction': round(float(prediction[0]), 2)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    if model is None or pipeline is None:
        print("\n" + "="*60)
        print("WARNING: Model or pipeline not found!")
        print("Please run 'python project.py' first to train the model.")
        print("="*60 + "\n")
    
    # Get port from environment variable (Render provides this)
    import os
    port = int(os.environ.get('PORT', 10000))
    
    # Run with production settings
    app.run(host='0.0.0.0', port=port, debug=False)
