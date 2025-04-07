import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# -- SHL assessments
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

# -- Load model from local directory
@st.cache_resource
def load_model():
    return SentenceTransformer('./all-MiniLM-L6-v2')

model = load_model()
assessment_embeddings = model.encode([a["description"] for a in assessments], convert_to_tensor=True)

# -- Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.title("📘 SHL Assessment Recommender")

input_method = st.radio("Select input method:", ["Enter Text", "Enter URL"])

query_text = ""

# -- Handle URL or manual input
if input_method == "Enter Text":
    query_text = st.text_area("Enter job description or related query:")
elif input_method == "Enter URL":
    url_input = st.text_input("Paste the job description URL:")

    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    if url_input and is_valid_url(url_input):
        try:
            response = requests.get(url_input, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            query_text = ' '.join(p.text for p in soup.find_all('p')[:5])
            st.success("Content extracted successfully from URL.")
        except Exception as e:
            st.error(f"Failed to fetch or parse URL: {e}")

if st.button("🔍 Recommend Assessments") and query_text.strip():
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, assessment_embeddings)[0]
    threshold = 0.4
    top_indices = [i for i in np.argsort(-similarities) if similarities[i] > threshold][:10]

    st.subheader("📝 Recommended SHL Assessments:")
    if top_indices:
        for idx in top_indices:
            a = assessments[idx]
            st.markdown(f"""
**{a['name']}**  
🔗 [Link to test]({a['url']})  
📄 Type: {a['test_type']}  
⏱ Duration: {a['duration']}  
🌐 Remote Testing: {a['remote_testing']}  
🧠 Adaptive IRT: {a['adaptive_irt']}  
📊 **Similarity Score**: {similarities[idx]:.2f}
---
""")
    else:
        st.warning("No matching assessments found. Try a more detailed input.")
