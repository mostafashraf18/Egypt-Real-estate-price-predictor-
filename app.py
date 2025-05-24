import joblib
import numpy as np
import logging
import os
from flask import Flask, request, jsonify, render_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_estate_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('real_estate_predictor')

app = Flask(__name__)

# Load the trained model and columns
logger.info("Loading model and columns...")
try:
    model = joblib.load('Real_Estate.pkl')
    columns = joblib.load('columns.pkl')
    logger.info("Model and columns loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or columns: {str(e)}")
    raise

# Extract locations and property types correctly
def extract_categories(columns, prefix):
    """Extract unique categories from columns with given prefix"""
    logger.info(f"Extracting categories with prefix: {prefix}")
    categories = set()
    for col in columns:
        if col.startswith(prefix):
            # Remove prefix and clean up the name
            name = col[len(prefix):].replace('_', ' ').strip()
            # Special handling for common cases
            if '6th of october' in name.lower():
                name = '6th of October, Giza'
            elif 'sheikh zayed' in name.lower():
                name = 'Sheikh Zayed, Giza'
            categories.add(name)
    return sorted(categories)

# Get locations and property types
locations = extract_categories(columns, 'location_')
property_types = extract_categories(columns, 'property_type_')

logger.info(f"📍 Locations found: {locations}")
logger.info(f"🏠 Property types found: {property_types}")

@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('index.html', locations=locations, types=property_types)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    try:
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        # Input validation
        location = data['location']
        beds = int(data['beds'])
        baths = int(data['baths'])
        area = float(data['area'])
        prop_type = data['type']
        
        logger.info(f"Processing request - Location: {location}, Beds: {beds}, Baths: {baths}, Area: {area}, Type: {prop_type}")

        if location not in locations:
            logger.warning(f"Invalid location: {location}")
            return jsonify({'error': f"Invalid location: {location}"}), 400
        if prop_type not in property_types:
            logger.warning(f"Invalid property type: {prop_type}")
            return jsonify({'error': f"Invalid property type: {prop_type}"}), 400
        if beds < 1 or baths < 1 or area < 50:
            logger.warning(f"Invalid numerical values: beds={beds}, baths={baths}, area={area}")
            return jsonify({'error': "Invalid numerical values"}), 400

        # Create input array
        x = np.zeros(len(columns))
        x[0] = beds
        x[1] = baths
        x[2] = area

        # Find matching columns (case-insensitive)
        loc_col = next((col for col in columns 
                      if col.startswith('location_') 
                      and location.lower() in col.lower()), None)
        
        type_col = next((col for col in columns 
                       if col.startswith('property_type_') 
                       and prop_type.lower() in col.lower()), None)

        logger.info(f"Matched location column: {loc_col}")
        logger.info(f"Matched property type column: {type_col}")

        if not loc_col or not type_col:
            logger.error("Category matching failed")
            return jsonify({'error': "Category matching failed"}), 400

        x[columns.get_loc(loc_col)] = 1
        x[columns.get_loc(type_col)] = 1

        # Make prediction
        logger.info("Making prediction")
        predicted_price = model.predict(x.reshape(1, -1))[0]
        formatted_price = f"EGP {predicted_price/1000000:.2f} Million" if predicted_price >= 1000000 else f"EGP {predicted_price:,.2f}"
        
        logger.info(f"Prediction result: {formatted_price}")
        return jsonify({'predicted_price': formatted_price})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting application server")
    # Make sure templates folder exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        logger.info("Created templates directory")
    
    # Make sure static folder exists
    if not os.path.exists('static'):
        os.makedirs('static')
        logger.info("Created static directory")
        
    app.run(debug=True, host='0.0.0.0', port=5000)