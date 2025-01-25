from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model.pkl')
model = joblib.load(model_path)

VEHICLE_MAP = {'Space Shuttle': 0, 'Space Taxi': 1, 'Space Pod': 2}

VEHICLE_EFFICIENCY = {
    'Space Shuttle': {
        'eco_rating': 85,
        'capacity_score': 95,
        'comfort_score': 90
    },
    'Space Taxi': {
        'eco_rating': 90,
        'capacity_score': 85,
        'comfort_score': 85
    },
    'Space Pod': {
        'eco_rating': 95,
        'capacity_score': 80,
        'comfort_score': 80
    }
}

def get_recommendations(emission, vehicle_type='Space Shuttle'):
    vehicle = VEHICLE_EFFICIENCY[vehicle_type]
    eco_rating = vehicle['eco_rating']
    
    if emission > 1000:
        return [
            f"💫 Your vehicle has an impressive {eco_rating}% eco-rating!",
            "✨ Experience cutting-edge space technology",
            "🌍 Leading the future of space exploration"
        ]
    elif emission > 500:
        return [
            f"⭐ This journey showcases our {eco_rating}% efficiency rating",
            "🌿 Pioneering sustainable space travel",
            "🎯 Setting new standards in space exploration"
        ]
    return [
        f"🏆 Maximum efficiency with {eco_rating}% eco-rating",
        "💫 Optimal space travel performance",
        "✨ Perfect choice for sustainable exploration"
    ]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        vehicle_type = data['vehicle_type']
        distance = data['distance']
        passengers = data['passengers']
        
        prediction = model.predict([[
            distance,
            VEHICLE_MAP[vehicle_type],
            passengers
        ]])[0]
        
        # Calculate eco metrics
        efficiency_score = max(0, 100 - (prediction/20))
        rewards = int(max(0, 1000 - prediction)/100)
        vehicle_stats = VEHICLE_EFFICIENCY[vehicle_type]
        
        return jsonify({
            'emission': round(prediction, 2),
            'unit': 'million tons CO2',
            'recommendations': get_recommendations(prediction, vehicle_type),
            'eco_score': round(efficiency_score, 1),
            'rewards_earned': rewards,
            'vehicle_efficiency': {
                'eco_rating': vehicle_stats['eco_rating'],
                'capacity_score': vehicle_stats['capacity_score'],
                'comfort_score': vehicle_stats['comfort_score']
            },
            'journey_stats': {
                'efficiency_level': 'Excellent' if efficiency_score > 80 else 'Good' if efficiency_score > 60 else 'Standard',
                'eco_impact': round((1000-prediction)/10, 1),
                'sustainability_score': round((vehicle_stats['eco_rating'] + efficiency_score)/2, 1)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)