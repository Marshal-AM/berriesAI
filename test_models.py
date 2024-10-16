# test_models.py

import joblib
import numpy as np

def test_credibility_model():
    try:
        credibility_model = joblib.load('credibility_model.joblib')
        print("Credibility Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Credibility Model: {e}")
        return

    # Test input
    test_input = np.array([[75000, 150000]])
    try:
        prediction = credibility_model.predict(test_input)
        print(f"Credibility Prediction: {prediction}")
    except Exception as e:
        print(f"Error predicting Credibility: {e}")

def test_influencer_model():
    try:
        influencer_model = joblib.load('influencer_model.joblib')
        print("Influencer Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Influencer Model: {e}")
        return

    # Test input (including credibility_weight)
    test_input = np.array([[15000, 750, 75, 7, 25, 3, 0.90]])
    try:
        prediction = influencer_model.predict(test_input)
        print(f"Influencer Prediction: {prediction}")
    except Exception as e:
        print(f"Error predicting Influencer Score: {e}")

if __name__ == "__main__":
    print("Testing Credibility Model:")
    test_credibility_model()
    print("\nTesting Influencer Model:")
    test_influencer_model()
