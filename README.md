#SHL Assessment Recommender System

A smart tool that recommends SHL assessments based on job descriptions or queries using Natural Language Processing.

---

##Project Links

- **Web App (Frontend - Streamlit):**  
  [https://shl-recommender-lw5hlpufzf7gckz5zhvcie.streamlit.app/](https://shl-recommender-lw5hlpufzf7gckz5zhvcie.streamlit.app/)

- **Backend API (FastAPI):**  
  [https://shl-recommender-ucpw.onrender.com/recommend](https://shl-recommender-ucpw.onrender.com/recommend)

---

##Features

- Accepts job descriptions or keywords
- Uses sentence embeddings (MiniLM model)
- Recommends relevant SHL assessments
- Accessible via Streamlit web interface
- REST API powered by FastAPI

---

##Tech Stack

| Layer       | Technology              |
|-------------|--------------------------|
| Frontend    | Streamlit                |
| Backend     | FastAPI + Uvicorn        |
| Model       | Sentence Transformers    |
| Embeddings  | all-MiniLM-L6-v2         |
| Hosting     | Render (backend) + Streamlit Cloud (frontend) |

---

##API Endpoint Usage

### POST `/recommend`

**Request Body:**
```json
{
  "query": "sales and communication skills"
}

Response:

{
  "results": [
    {
      "name": "Customer Service Simulation",
      "url": "https://www.shl.com/product/customer-service-simulation/",
      "remote_testing": "Yes",
      "adaptive_irt": "Yes",
      "duration": "35 minutes",
      "test_type": "Simulation",
      "similarity_score": 0.72
    },
    ...
  ]
}
