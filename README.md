# HireMatch AI — CV & Job Description Analyzer

An AI-powered CV analysis tool that tells you exactly 
how well your CV matches a job description and what 
to fix before you apply.

## Live Demo
(https://hirematch-aii.streamlit.app/)

## What it does
- Calculates a match score between your CV and any job description
- Breaks down 3 sub-scores: Skills, Experience, Keywords
- Shows exact matched skills, semantically similar skills, 
  and truly missing skills
- Keyword gap analysis chart — see which JD keywords 
  are missing from your CV
- Skill similarity heatmap — visual overlap between 
  your skills and the job requirements
- ATS compatibility check — will your CV pass automated screening?
- Specific action plan — exactly what to fix, not generic advice

## How it works
- Sentence transformers (all-MiniLM-L6-v2) for semantic 
  similarity — understands meaning not just keywords
- Cosine similarity for overall CV-JD match scoring
- TF-IDF for keyword frequency analysis
- Pairwise skill similarity heatmap using embeddings

## Tech stack
- Python
- Streamlit
- Sentence Transformers
- Scikit-learn
- Plotly
- Pandas

## Run locally
git clone https://github.com/YOUR_USERNAME/hirematch-ai
cd hirematch-ai
pip install -r requirements.txt
streamlit run app.py

## Screenshots
(add screenshots here after deploying)
