import streamlit as st
import requests

# Set your FastAPI backend URL here
API_URL = "https://shl-recommender-ucpw.onrender.com/recommend"

st.title("SHL Assessment Recommender")
st.markdown("Enter a job description or query, and get recommended assessments.")

# Input box for query
query = st.text_area("Enter Job Description / Query", height=200)

if st.button("Get Recommendations"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(API_URL, json={"query": query})
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("recommended_assessments", [])
                    if results:
                        st.success(f"{len(results)} Recommendations found!")
                        for rec in results:
                            st.write("----")
                            st.markdown(f"**URL:** [{rec['url']}]({rec['url']})")
                            st.markdown(f"**Adaptive Support:** {rec['adaptive_support']}")
                            st.markdown(f"**Remote Support:** {rec['remote_support']}")
                            st.markdown(f"**Duration:** {rec['duration']} minutes")
                            st.markdown(f"**Test Type:** {', '.join(rec['test_type'])}")
                            st.markdown(f"**Description:** {rec['description']}")
                    else:
                        st.info("No recommendations found.")
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to fetch recommendations: {e}")
