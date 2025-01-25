import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

def generate_training_data():
    X = []
    y = []
    vehicles = {
        'Space Shuttle': {'efficiency': 0.7, 'base_emission': 1000},
        'Space Taxi': {'efficiency': 0.8, 'base_emission': 500},
        'Space Pod': {'efficiency': 0.9, 'base_emission': 250}
    }
    
    for _ in range(1000):
        distance = np.random.uniform(400, 225000000)
        vehicle_type = np.random.choice([0, 1, 2])
        passengers = np.random.randint(1, 20)
        X.append([distance, vehicle_type, passengers])
        vehicle = list(vehicles.values())[vehicle_type]
        emission = (distance * vehicle['base_emission'] * (1 - vehicle['efficiency']) * 
                   (1 + passengers * 0.1)) / 1000000
        y.append(emission)
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = generate_training_data()
    model = LinearRegression()
    model.fit(X, y)
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    joblib.dump(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl'))