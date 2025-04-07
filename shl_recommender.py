import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Load model and assessments
model = SentenceTransformer('all-MiniLM-L6-v2')

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

assessment_texts = [a["description"] for a in assessments]
assessment_embeddings = model.encode(assessment_texts, convert_to_tensor=True)

# Helper function
def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.text for p in paragraphs[:5])
    except Exception as e:
        return f"Error fetching data: {str(e)}"

def get_recommendations(query_text):
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
    return results

# Streamlit UI
st.title("🧠 SHL Assessment Recommender")

option = st.radio("Choose Input Method:", ("Job Description Text", "Job URL"))

input_text = ""

if option == "Job Description Text":
    input_text = st.text_area("Paste the job description here")
else:
    url = st.text_input("Enter job description URL:")
    if url and is_valid_url(url):
        input_text = get_text_from_url(url)
        st.info("Fetched text from URL:")
        st.write(input_text)

if st.button("Recommend Assessments") and input_text.strip():
    recommendations = get_recommendations(input_text)
    if recommendations:
        st.success("Recommended Assessments:")
        for rec in recommendations:
            st.write(f"**{rec['name']}** - [{rec['url']}]({rec['url']})")
            st.write(f"Type: {rec['test_type']} | Duration: {rec['duration']}")
            st.write(f"Remote: {rec['remote_testing']} | Adaptive IRT: {rec['adaptive_irt']}")
            st.write(f"Similarity Score: {rec['similarity_score']:.2f}")
            st.markdown("---")
    else:
        st.warning("No relevant assessments found.")
