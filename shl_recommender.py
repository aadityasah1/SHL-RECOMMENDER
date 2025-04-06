# SHL Assessment Recommender System
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging


logging.basicConfig(level=logging.INFO)

#Load SHL assessments
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

#Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')

assessment_texts = [a["description"] for a in assessments]
assessment_embeddings = model.encode(assessment_texts, convert_to_tensor=True)

#FastApi initialization
app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0",
    description="Recommends SHL assessments based on job description or query."
)

#Input Factor
class QueryInput(BaseModel):
    query: str = None
    url: str = None

#Url Checking
def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

#Endpoint Recommendation
@app.post("/recommend")
async def recommend(data: QueryInput):
    logging.info(f"Incoming request: {data}")

    # Extract query from URL or direct input
    query_text = ""
    if data.url:
        if not is_valid_url(data.url):
            return {"error": "Invalid URL provided"}

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

    results = []
    for idx in top_indices:
        a = assessments[idx]
        results.append({
            "name": a["name"],
            "url": a["url"],
            "remote_testing": a["remote_testing"],
            "adaptive_irt": a["adaptive_irt"],
            "duration": a["duration"],
            "test_type": a["test_type"],
            "similarity_score": float(similarities[idx])
        })

    return {"results": results}

#Run Locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
