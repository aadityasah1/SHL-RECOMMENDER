import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

# Sample SHL assessments database
assessments = [
    {
        "name": "General Ability Test",
        "url": "https://www.shl.com/product/general-ability-test/",
        "description": "Measures numerical, verbal, and logical reasoning abilities.",
        "remote_testing": "Yes",
        "adaptive_irt": "Yes",
        "duration": "30 minutes",
        "test_type": "Cognitive"
    },
    {
        "name": "Sales Personality Questionnaire",
        "url": "https://www.shl.com/product/sales-personality-questionnaire/",
        "description": "Assesses personality traits important for success in sales roles.",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "25 minutes",
        "test_type": "Personality"
    },
    {
        "name": "Customer Service Simulation",
        "url": "https://www.shl.com/product/customer-service-simulation/",
        "description": "Simulates real-world scenarios to evaluate customer service skills.",
        "remote_testing": "Yes",
        "adaptive_irt": "Yes",
        "duration": "35 minutes",
        "test_type": "Simulation"
    }
]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute embeddings
assessment_texts = [a["description"] for a in assessments]
assessment_embeddings = model.encode(assessment_texts, convert_to_tensor=True)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0",
    description="Recommends SHL assessments based on job description or query."
)

# Input model
class QueryInput(BaseModel):
    query: str = None
    url: str = None

# Utility: Validate URL
def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Health check endpoint for Render
@app.get("/")
async def root():
    return {"message": "SHL Recommender API is live!"}

# POST-only endpoint for recommendations
@app.post("/recommend")
async def recommend(data: QueryInput):
    logging.info(f"Incoming request: {data}")

    query_text = ""

    if data.url:
        if not is_valid_url(data.url):
            return {"error": "Invalid URL provided."}
        try:
            response = requests.get(data.url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            query_text = ' '.join(p.text for p in paragraphs[:5])
        except requests.RequestException as e:
            return {"error": f"Failed to fetch or parse URL: {str(e)}"}
    elif data.query:
        query_text = data.query
    else:
        return {"error": "No query or URL provided."}

    if not query_text.strip():
        return {"error": "Extracted text is empty."}

    # Compute similarity
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, assessment_embeddings)[0]

    threshold = 0.4
    top_indices = [i for i in np.argsort(-similarities) if similarities[i] > threshold][:10]

    recommended_assessments = [
        {
            "name": assessments[idx]["name"],
            "url": assessments[idx]["url"],
            "remote_support": assessments[idx]["remote_testing"],
            "adaptive_support": assessments[idx]["adaptive_irt"],
            "duration": assessments[idx]["duration"].replace(" minutes", ""),
            "test_type": [assessments[idx]["test_type"]],
            "description": assessments[idx]["description"]
        }
        for idx in top_indices
    ]

    return {"recommended_assessments": recommended_assessments}

# Optional: catch incorrect GET usage on /recommend
@app.get("/recommend")
async def recommend_get():
    return {"error": "Use POST method with a query or URL."}

# Log on startup
@app.on_event("startup")
def startup_event():
    logging.info("✅ SHL Recommender API started successfully!")

# Run locally or on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
