import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Load SHL assessments
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

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
assessment_embeddings = model.encode([a["description"] for a in assessments], convert_to_tensor=True)

# Streamlit UI
st.title("🔍 SHL Assessment Recommender")

option = st.radio("Choose input method:", ("Enter text", "Paste job URL"))

if option == "Enter text":
    query = st.text_area("Enter job description or role:")
elif option == "Paste job URL":
    url = st.text_input("Enter job description URL:")
    if url:
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            query = ' '.join(p.text for p in soup.find_all('p')[:5])
            st.success("Extracted text from URL.")
        except:
            st.error("Failed to extract content from URL.")
            query = ""

if st.button("Recommend Assessments") and query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, assessment_embeddings)[0]
    threshold = 0.4
    top_indices = [i for i in np.argsort(-similarities) if similarities[i] > threshold][:10]

    st.subheader("Recommended Assessments:")
    for idx in top_indices:
        a = assessments[idx]
        st.markdown(f"**{a['name']}**  \nURL: {a['url']}  \nType: {a['test_type']}  \nRemote Testing: {a['remote_testing']}  \nAdaptive IRT: {a['adaptive_irt']}  \nDuration: {a['duration']}  \n**Similarity**: {similarities[idx]:.2f}")
        st.markdown("---")
