import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SocioBerries API with External Data Fetching",
    description="API to calculate Credibility Weight and Influencer Score (Berries) using external data",
    version="1.0.0"
)

# Load the ML models at startup
try:
    credibility_model = joblib.load('credibility_model.joblib')
    influencer_model = joblib.load('influencer_model.joblib')
    logger.info("Models loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise e
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e

class BerryCalculationInput(BaseModel):
    account_address: str

class BerryCalculationOutput(BaseModel):
    credibility_weight: float
    berries: float

async def fetch_user_posts(address: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://resources-bfih.onrender.com/get-user-posts?AccountAddress={address}")
        response.raise_for_status()
        return response.json()

async def fetch_coin_balances(address: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(f"https://balance-7m39.onrender.com/coin_balances?address={address}")
        response.raise_for_status()
        return response.json()

def calculate_credibility_weight(onchain_net_worth: float, followers_of_followers: int):
    features = np.array([[onchain_net_worth, followers_of_followers]])
    prediction = credibility_model.predict(features)
    return prediction[0] if prediction.ndim == 1 else prediction[0][0]

def calculate_berries(input_data: dict):
    features = np.array([[
        input_data['followers'],
        input_data['likes'],
        input_data['comments'],
        input_data['ads_purchased_from_profile'],
        input_data['average_likes_per_day'],
        input_data['average_comments_per_day'],
        input_data['credibility_weight']
    ]])
    prediction = influencer_model.predict(features)
    return prediction[0] if prediction.ndim == 1 else prediction[0][0]

@app.post("/calculate_berries", response_model=BerryCalculationOutput)
async def calculate_berries_endpoint(input: BerryCalculationInput):
    try:
        # Hardcoded values
        followers_of_followers = 5000
        ads_purchased_from_profile = 10

        # Fetch user posts data
        user_posts_data = await fetch_user_posts(input.account_address)
        
        # Extract total followers, likes, comments, and post count
        total_followers = 0
        total_likes = 0
        total_comments = 0
        total_posts = 0
        
        for resource in user_posts_data['resources']:
            if resource['type'].endswith('::SocialMediaPlatform::Profile'):
                total_followers += int(resource['data']['followers_count'])
            elif resource['type'].endswith('::SocialMediaPlatform::UserPosts'):
                for post in resource['data']['posts']:
                    total_likes += int(post['like_count'])
                    total_comments += len(post['comments'])
                    total_posts += 1

        # Calculate average likes and comments per day
        average_likes_per_day = total_likes / total_posts if total_posts > 0 else 0
        average_comments_per_day = total_comments / total_posts if total_posts > 0 else 0

        # Fetch coin balances
        coin_balances = await fetch_coin_balances(input.account_address)
        
        # Extract onchain net worth (amount in USD)
        onchain_net_worth = 0
        for balance in coin_balances['latest_balances']:
            if balance['metadata']['symbol'] == 'APT':
                onchain_net_worth = balance.get('amount_in_usd', 0)
                break

        # Calculate credibility weight
        credibility_weight = calculate_credibility_weight(onchain_net_worth, followers_of_followers)

        # Prepare input for berries calculation
        berries_input = {
            'followers': total_followers,
            'likes': total_likes,
            'comments': total_comments,
            'ads_purchased_from_profile': ads_purchased_from_profile,
            'average_likes_per_day': average_likes_per_day,
            'average_comments_per_day': average_comments_per_day,
            'credibility_weight': credibility_weight
        }

        # Calculate berries
        berries = calculate_berries(berries_input)

        return BerryCalculationOutput(
            credibility_weight=credibility_weight,
            berries=berries
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching external data: {str(e)}")
    except KeyError as e:
        logger.error(f"KeyError: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected data structure in external API response: {str(e)}")
    except Exception as e:
        logger.exception(f"Error in calculate_berries_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)