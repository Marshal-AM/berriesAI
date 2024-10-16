# Importing necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI instance
app = FastAPI()

# Define request and response models

# Model for the credibility prediction input
class CredibilityInput(BaseModel):
    selective_liking: float
    selective_commenting: float
    balance_in_dollars: float
    social_status: float
    quality_of_followers: float

# Model for the credibility prediction output
class CredibilityOutput(BaseModel):
    credibility_weight: float
    correlation_factor: float

# Model for the influencer prediction input
class InfluencerInput(BaseModel):
    follower_count: int
    likes: int
    comments: int
    ad_products_sold: int
    likes_frequency: float
    comments_frequency: float
    credibility_weight: float  # This is the input from the credibility model

# Model for the influencer prediction output
class InfluencerOutput(BaseModel):
    influencer_score: float
    correlation_factor: float

# Dummy models for prediction (replace with actual models)
class CredibilityModel:
    @staticmethod
    def predict(data):
        # Placeholder prediction logic for credibility model
        # Replace this with your actual model logic
        return [0.75]  # Example credibility weight

class InfluencerModel:
    @staticmethod
    def predict(data):
        # Placeholder prediction logic for influencer model
        # Replace this with your actual model logic
        return [85.0]  # Example influencer score

# Initialize models
credibility_model = CredibilityModel()
influencer_model = InfluencerModel()

# Endpoint to predict Credibility Weight
@app.post("/predict_credibility", response_model=CredibilityOutput)
def predict_credibility(input: CredibilityInput):
    logger.info(f"Received credibility prediction request: {input}")
    try:
        # Prepare the input data for the model
        data = np.array([
            input.selective_liking,
            input.selective_commenting,
            input.balance_in_dollars,
            input.social_status,
            input.quality_of_followers
        ]).reshape(1, -1)
        
        # Predict Credibility Weight
        credibility_weight = credibility_model.predict(data)[0]
        correlation_factor = 0.85  # Example correlation factor
        logger.info(f"Credibility weight predicted: {credibility_weight}")

        return CredibilityOutput(
            credibility_weight=credibility_weight,
            correlation_factor=correlation_factor
        )
    except Exception as e:
        logger.error(f"Error in /predict_credibility: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Endpoint to predict Influencer Score (Berries)
@app.post("/predict_influencer", response_model=InfluencerOutput)
def predict_influencer(input: InfluencerInput):
    logger.info(f"Received influencer prediction request: {input}")
    try:
        # Prepare the input data for the model
        data = np.array([
            input.follower_count,
            input.likes,
            input.comments,
            input.ad_products_sold,
            input.likes_frequency,
            input.comments_frequency,
            input.credibility_weight  # Use credibility weight from previous prediction
        ]).reshape(1, -1)

        # Predict Influencer Score
        influencer_score = influencer_model.predict(data)[0]
        correlation_factor = 0.90  # Example correlation factor
        logger.info(f"Influencer score predicted: {influencer_score}")

        return InfluencerOutput(
            influencer_score=influencer_score,
            correlation_factor=correlation_factor
        )
    except Exception as e:
        logger.error(f"Error in /predict_influencer: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# To run the app: uvicorn app.main:app --reload (in terminal)
