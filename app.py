"""
app.py — HireMatch AI: dark-mode premium CV screener UI.

Two-step flow:
  Step 1 (input)   — paste/upload CV + paste JD, optional demo data
  Step 2 (results) — full analysis: score, skills, ATS, chart, action plan, verdict
"""

import io
import html as html_lib
import streamlit as st
import plotly.graph_objects as go
from screener import analyze

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HireMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "bg":          "#0F0F1A",
    "card":        "#1A1A2E",
    "card2":       "#14142A",
    "border":      "rgba(108,99,255,0.22)",
    "border_hi":   "rgba(108,99,255,0.55)",
    "purple":      "#6C63FF",
    "purple_dim":  "#A5B4FC",
    "text":        "#F1F5F9",
    "text2":       "#94A3B8",
    "text3":       "#4B5563",
    "divider":     "#1E1E35",
    "track":       "#2D2D48",
    "green":       "#4ADE80",
    "green_bg":    "#0A2B1A",
    "green_bdr":   "#14532D",
    "amber":       "#FCD34D",
    "amber_bg":    "#1C1600",
    "red":         "#F87171",
    "red_bg":      "#2C0A0A",
    "red_bdr":     "#7F1D1D",
    "blue":        "#60A5FA",
    "blue_bg":     "#070E1C",
    "blue_bdr":    "#1E3A5F",
}

# ── Demo content ───────────────────────────────────────────────────────────────
SAMPLE_CV = """Jane Doe
jane.doe@email.com | github.com/janedoe | linkedin.com/in/janedoe | +1 555 0123

PROFESSIONAL SUMMARY
Data Engineer with 4 years of experience building production-grade data pipelines and
deploying ML models to AWS. Strong Python background with hands-on experience in
containerisation and orchestration. Eager to grow into a full ML Engineering role.

TECHNICAL SKILLS
Python, SQL, PostgreSQL, Docker, AWS (S3, EC2, Lambda, RDS), Git, Apache Spark,
Airflow, Pandas, NumPy, Scikit-learn, Flask, REST APIs, Linux, Bash, Tableau

EXPERIENCE
Senior Data Engineer | DataFlow Inc | 2021–Present
- Built and maintained ETL pipelines processing 5M+ records daily using Python and Apache Spark
- Deployed 3 scikit-learn models to production via Docker and Flask REST APIs on AWS
- Reduced data processing latency by 65% through pipeline optimisation and caching
- Mentored 2 junior engineers and led weekly code reviews for a team of 6

Data Engineer | Analytics Co | 2019–2021
- Developed data ingestion pipelines in Python, processing 500K records/day
- Designed PostgreSQL schemas for a multi-tenant SaaS product serving 10K users
- Built Tableau dashboards adopted by 5 business teams
- Worked in an Agile/Scrum environment with 2-week sprints

EDUCATION
BSc Data Science | University of Edinburgh | 2015–2019 | First-Class Honours

PROJECTS
- Fraud detection model (scikit-learn, 92% accuracy) deployed on AWS Lambda
- Open-source Airflow plugin for data quality checks (200+ GitHub stars)
"""

SAMPLE_JD = """Machine Learning Engineer | FinTech AI Startup

We are building AI-powered financial tools and need an ML Engineer to take our models
from research to production at scale.

REQUIREMENTS
- 3+ years in machine learning or MLOps engineering
- Proficient in Python and deep learning frameworks (PyTorch or TensorFlow)
- Experience deploying models with Kubernetes and Docker
- Strong NLP background — transformer models, BERT, Hugging Face ecosystem
- SQL and NoSQL (MongoDB) database experience
- MLflow or equivalent experiment tracking tools
- Cloud platforms: GCP or AWS
- Understanding of LLM fine-tuning and prompt engineering

RESPONSIBILITIES
- Train and deploy NLP models for transaction classification and fraud detection
- Build MLOps pipelines for model versioning, monitoring, and automated retraining
- Collaborate with data engineers on feature engineering pipelines
- Optimise model inference latency for real-time scoring (target < 100ms)
- Work in an agile team alongside product, research, and data engineering

NICE TO HAVE
- Experience with Kubeflow or Vertex AI
- Kaggle competition history or published ML research
- Distributed training (PyTorch Lightning, DeepSpeed)
"""


# ── PDF text extraction ────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
        return "\n\n".join(pages).strip()
    except ImportError:
        st.error("Install pdfplumber (`pip install pdfplumber`) to enable PDF upload.")
        return ""
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return ""


# ── SVG score ring ─────────────────────────────────────────────────────────────
def score_ring(score: float, color: str, track: str) -> str:
    r    = 62
    cx   = cy = 76
    circ = 2 * 3.14159265 * r
    fill = circ * score / 100
    gap  = circ - fill
    return f"""
    <svg width="152" height="152" viewBox="0 0 152 152"
         style="display:block;margin:0 auto;">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="{track}" stroke-width="11"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="{color}" stroke-width="11" stroke-linecap="round"
              stroke-dasharray="{fill:.1f} {gap:.1f}"
              transform="rotate(-90 {cx} {cy})"/>
      <text x="{cx}" y="{cy - 7}" text-anchor="middle"
            font-family="Inter,sans-serif" font-size="28"
            font-weight="800" fill="#FFFFFF">{score}%</text>
      <text x="{cx}" y="{cy + 14}" text-anchor="middle"
            font-family="Inter,sans-serif" font-size="11"
            font-weight="600" fill="#6B7280" letter-spacing="1">MATCH</text>
    </svg>"""


# ── Sub-score progress bar ─────────────────────────────────────────────────────
def sub_score_bar(label: str, value: float, color: str) -> str:
    w = max(2, min(100, value))
    return f"""
    <div style="margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span style="font-size:13px;font-weight:500;color:{C['text2']};">{label}</span>
        <span style="font-size:13px;font-weight:700;color:{color};">{value:.0f}%</span>
      </div>
      <div style="height:6px;background:{C['track']};border-radius:999px;overflow:hidden;">
        <div style="height:100%;width:{w}%;background:{color};border-radius:999px;"></div>
      </div>
    </div>"""


# ── Keyword gap chart ─────────────────────────────────────────────────────────
def keyword_chart(kw_data: list[dict]):
    if not kw_data:
        st.info("Not enough keyword data to render this chart.")
        return

    labels    = [k["keyword"] for k in kw_data][::-1]
    jd_counts = [k["jd_count"] for k in kw_data][::-1]
    cv_counts = [k["cv_count"] for k in kw_data][::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=jd_counts, name="Job Description", orientation="h",
        marker=dict(color="rgba(96,165,250,0.18)",
                    line=dict(color="rgba(96,165,250,0.70)", width=1)),
    ))
    fig.add_trace(go.Bar(
        y=labels, x=cv_counts, name="Your CV", orientation="h",
        marker=dict(color="#F97316"),
    ))
    fig.update_layout(
        barmode="overlay",
        title=dict(text="Keyword Gap Analysis",
                   font=dict(size=14, color=C["text"], family="Inter, sans-serif"), x=0),
        plot_bgcolor=C["card"], paper_bgcolor=C["card"],
        font=dict(family="Inter, sans-serif", size=12, color=C["text"]),
        margin=dict(l=0, r=16, t=44, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1,
                    font=dict(size=12, color=C["text2"]),
                    bgcolor="rgba(0,0,0,0)"),
        height=420,
        xaxis=dict(showgrid=True, gridcolor=C["track"], title="Occurrences",
                   tickfont=dict(size=11, color=C["text2"]),
                   color=C["text2"], zeroline=False),
        yaxis=dict(tickfont=dict(size=12, color=C["text"]), tickcolor=C["card"]),
        hoverlabel=dict(bgcolor=C["card2"], font_size=12, font_color=C["text"]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Skill similarity heatmap ───────────────────────────────────────────────────
def skill_heatmap_chart(heatmap_data: dict):
    """Pairwise cosine similarity heatmap: JD skills (rows) vs CV skills (cols)."""
    if not heatmap_data.get("matrix"):
        return

    jd_skills = heatmap_data["jd_skills"]
    cv_skills = heatmap_data["cv_skills"]
    matrix    = heatmap_data["matrix"]

    def short(s): return s[:16] + "…" if len(s) > 16 else s

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[short(s) for s in cv_skills],
        y=[short(s) for s in jd_skills],
        colorscale=[
            [0.00, "#0D0D1A"],
            [0.30, "#1A1A2E"],
            [0.60, "#3D3580"],
            [1.00, "#6C63FF"],
        ],
        zmin=0, zmax=1,
        hoverongaps=False,
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=10, color="#FFFFFF"),
        colorbar=dict(
            title=dict(text="Similarity", font=dict(color=C["text2"], size=11)),
            tickfont=dict(color=C["text2"], size=10),
            outlinewidth=0,
            thickness=14,
        ),
    ))
    fig.update_layout(
        plot_bgcolor=C["card2"], paper_bgcolor=C["card2"],
        font=dict(family="Inter, sans-serif", size=11, color=C["text"]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        xaxis=dict(
            title=dict(text="Your CV Skills", font=dict(color=C["text2"])),
            tickfont=dict(size=10, color=C["text"]),
            tickangle=-30,
            color=C["text2"],
        ),
        yaxis=dict(
            title=dict(text="JD Required Skills", font=dict(color=C["text2"])),
            tickfont=dict(size=10, color=C["text"]),
            color=C["text2"],
            autorange="reversed",
        ),
        hoverlabel=dict(bgcolor=C["card2"], font_size=11, font_color=C["text"]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── HTML export report ─────────────────────────────────────────────────────────
def build_html_report(result: dict) -> str:
    r       = result
    score   = r["score"]
    verdict = "Strong Match" if score >= 70 else "Partial Match" if score >= 45 else "Weak Match"
    color   = "#4ADE80" if score >= 70 else "#FCD34D" if score >= 45 else "#F87171"

    def pills(items, bg, fg):
        return " ".join(
            f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;'
            f'margin:3px 2px;font-size:12px;background:{bg};color:{fg};">'
            f'{html_lib.escape(s)}</span>' for s in items
        ) or "<em style='color:#94A3B8;'>None</em>"

    def section(title, body):
        return (f'<div style="margin-bottom:28px;">'
                f'<h2 style="font-size:17px;font-weight:700;color:#F1F5F9;'
                f'border-bottom:2px solid #2D2D48;padding-bottom:8px;margin-bottom:14px;">'
                f'{html_lib.escape(title)}</h2>{body}</div>')

    skills_html = (
        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;'>"
        f"<div><strong style='color:#4ADE80;'>Matched ({len(r['matched'])})</strong><br>"
        f"{pills(r['matched'], '#0A2B1A', '#4ADE80')}</div>"
        f"<div><strong style='color:#60A5FA;'>Semantic ({len(r['semantic_matched'])})</strong><br>"
        f"{pills(list(r['semantic_matched'].keys()), '#070E1C', '#60A5FA')}</div>"
        f"<div><strong style='color:#F87171;'>Missing ({len(r['missing'])})</strong><br>"
        f"{pills(r['missing'], '#2C0A0A', '#F87171')}</div></div>"
    )

    def plan_section(items, border, bg, label):
        if not items:
            return ""
        items_html = "".join(
            f'<li style="margin-bottom:8px;color:#CBD5E1;">{html_lib.escape(i)}</li>'
            for i in items
        )
        return (f'<div style="border-left:4px solid {border};background:{bg};'
                f'border-radius:8px;padding:14px 18px;margin-bottom:12px;">'
                f'<strong style="color:{border};">{html_lib.escape(label)}</strong>'
                f'<ul style="margin:8px 0 0 16px;padding:0;">{items_html}</ul></div>')

    action_html = (
        plan_section(r["action_plan"]["critical"],    "#F87171", "#2C0A0A", "Priority 1 — Critical") +
        plan_section(r["action_plan"]["important"],   "#FCD34D", "#1C1600", "Priority 2 — Important") +
        plan_section(r["action_plan"]["nice_to_have"],"#60A5FA", "#070E1C", "Priority 3 — Nice to Have")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HireMatch AI Report</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:#0F0F1A;color:#F1F5F9;margin:0;padding:32px; }}
  .card {{ background:#1A1A2E;border:1px solid rgba(108,99,255,0.22);border-radius:12px;
           padding:24px 28px;margin-bottom:24px; }}
  h1 {{ font-size:26px;font-weight:800;color:#6C63FF;margin:0 0 4px; }}
  .tagline {{ color:#94A3B8;font-size:14px; }}
  @media print {{ body {{ background:#fff;color:#0F172A; }} }}
</style>
</head>
<body>
<div class="card"><h1>HireMatch AI</h1>
<p class="tagline">AI-powered CV Analysis Report</p></div>
<div class="card" style="text-align:center;">
  <div style="font-size:52px;font-weight:800;color:{color};">{score}%</div>
  <div style="font-size:18px;font-weight:600;color:{color};margin:4px 0 16px;">{html_lib.escape(verdict)}</div>
  <div style="display:flex;justify-content:center;gap:32px;">
    <div><strong style="font-size:20px;color:#4ADE80;">{len(r['matched'])+len(r['semantic_matched'])}</strong>
      <div style="font-size:12px;color:#94A3B8;">Matched</div></div>
    <div><strong style="font-size:20px;color:#F87171;">{len(r['missing'])}</strong>
      <div style="font-size:12px;color:#94A3B8;">Missing</div></div>
    <div><strong style="font-size:20px;color:#6C63FF;">{r['ats_score']:.0f}</strong>
      <div style="font-size:12px;color:#94A3B8;">ATS Score</div></div>
  </div>
</div>
<div class="card">
  {section("Sub-Scores",
    f"<span style='color:#94A3B8;'>Skills Match: <strong style='color:#F1F5F9;'>"
    f"{r['sub_scores']['skills_match']:.0f}%</strong> &nbsp;|&nbsp; "
    f"Experience Relevance: <strong style='color:#F1F5F9;'>{r['sub_scores']['experience_relevance']:.0f}%</strong>"
    f" &nbsp;|&nbsp; Keywords Match: <strong style='color:#F1F5F9;'>{r['sub_scores']['keywords_match']:.0f}%</strong></span>"
  )}
  {section("Skills Analysis", skills_html)}
  {section("Strengths","<ul style='margin:0;padding-left:20px;'>" +
    "".join(f"<li style='margin-bottom:6px;color:#CBD5E1;'>{html_lib.escape(s)}</li>" for s in r['strengths']) +
    "</ul>")}
  {section("Weaknesses","<ul style='margin:0;padding-left:20px;'>" +
    "".join(f"<li style='margin-bottom:6px;color:#CBD5E1;'>{html_lib.escape(s)}</li>" for s in r['weaknesses']) +
    "</ul>")}
  {section("Action Plan", action_html)}
  {section("Final Verdict",
    f"<p style='color:#CBD5E1;line-height:1.75;font-size:14px;'>{html_lib.escape(r['verdict'])}</p>"
  )}
</div>
<p style="text-align:center;color:#4B5563;font-size:12px;margin-top:24px;">
  Generated by HireMatch AI · Powered by all-MiniLM-L6-v2 sentence embeddings
</p>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Global CSS — dark premium theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset ── */
html, body, [class*="css"], .stApp {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background-color: {C['bg']} !important;
  color: {C['text']} !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {C['track']}; border-radius: 3px; }}

/* ── Navbar ── */
.hm-nav {{
  background: {C['bg']};
  border-bottom: 1px solid {C['divider']};
  padding: 0 48px; height: 60px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 999;
  box-shadow: 0 1px 0 {C['divider']};
}}
.hm-logo {{
  display: flex; align-items: center; gap: 10px;
  font-size: 19px; font-weight: 800; color: {C['text']}; letter-spacing: -.4px;
}}
.hm-logo-icon {{
  width: 32px; height: 32px; border-radius: 8px;
  background: {C['purple']}; display: flex; align-items: center;
  justify-content: center; font-size: 16px; flex-shrink: 0;
}}
.hm-logo-ai {{
  font-size: 11px; font-weight: 700; color: {C['purple']};
  background: rgba(108,99,255,0.14); padding: 2px 7px;
  border-radius: 4px; margin-left: 2px; letter-spacing: .3px;
}}
.hm-nav-steps {{
  display: flex; align-items: center; gap: 8px;
  font-size: 12.5px; color: {C['text2']}; font-weight: 500;
}}
.step-dot {{
  width: 22px; height: 22px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700;
}}
.step-active {{ background: {C['purple']}; color: #fff; }}
.step-done   {{ background: {C['green_bg']}; color: {C['green']}; }}
.step-idle   {{ background: {C['track']}; color: {C['text3']}; }}
.step-line   {{ width: 28px; height: 1px; background: {C['divider']}; }}
.hm-badge {{
  font-size: 11px; font-weight: 600; color: {C['purple_dim']};
  background: rgba(108,99,255,0.14); padding: 3px 10px; border-radius: 999px;
  letter-spacing: .3px;
}}

/* ── Hero ── */
.hm-hero {{
  background: linear-gradient(135deg, #1E1B4B 0%, #312E81 45%, #1E1B4B 100%);
  padding: 56px 48px 52px; text-align: center;
  border-bottom: 1px solid {C['divider']};
}}
.hm-hero h1 {{
  font-size: 36px; font-weight: 800; color: #fff !important;
  margin: 0 0 14px; letter-spacing: -.7px; line-height: 1.2;
}}
.hm-hero p {{
  font-size: 15.5px; color: rgba(255,255,255,.70); margin: 0; line-height: 1.65;
}}
.hm-hero-chips {{
  display: flex; justify-content: center; gap: 8px;
  margin-top: 20px; flex-wrap: wrap;
}}
.hm-chip {{
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.15);
  color: rgba(255,255,255,.80); font-size: 11.5px; font-weight: 500;
  padding: 4px 12px; border-radius: 999px;
}}

/* ── Content wrapper ── */
.hm-wrap {{ max-width: 1180px; margin: 0 auto; padding: 36px 40px 60px; }}

/* ── Input card columns ── */
.input-cols [data-testid="column"],
.input-cols [data-testid="stColumn"] {{
  background: {C['card']} !important;
  border: 1px solid {C['border']} !important;
  border-radius: 12px !important;
  padding: 22px 22px 16px !important;
  transition: border-color .2s;
}}
.input-cols [data-testid="column"]:focus-within,
.input-cols [data-testid="stColumn"]:focus-within {{
  border-color: {C['border_hi']} !important;
}}

/* Panel header inside card */
.panel-head {{
  display: flex; align-items: center; gap: 10px; margin-bottom: 14px;
}}
.panel-icon {{
  width: 32px; height: 32px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center; font-size: 16px;
  background: rgba(108,99,255,0.2);
}}
.panel-title {{ font-size: 14.5px; font-weight: 700; color: {C['text']}; }}
.panel-sub   {{ font-size: 11.5px; color: {C['text2']}; margin-top: 1px; }}
.char-count  {{
  font-size: 11px; color: {C['text3']}; text-align: right;
  margin-top: 6px; font-weight: 500; font-variant-numeric: tabular-nums;
}}

/* ── Textarea ── */
.stTextArea textarea {{
  background: {C['bg']} !important;
  border: 1px solid {C['divider']} !important;
  border-radius: 8px !important;
  color: {C['text']} !important;
  font-size: 13px !important;
  font-family: 'Inter', sans-serif !important;
  line-height: 1.65 !important;
  padding: 12px 14px !important;
  box-shadow: none !important;
  resize: vertical !important;
  caret-color: {C['purple']};
}}
.stTextArea textarea:focus {{
  border-color: {C['purple']} !important;
  box-shadow: 0 0 0 3px rgba(108,99,255,.18) !important;
  outline: none !important;
}}
.stTextArea textarea::placeholder {{ color: {C['track']} !important; }}
.stTextArea label {{ display: none !important; }}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
  border: 1px dashed rgba(108,99,255,0.35) !important;
  border-radius: 8px !important;
  background: rgba(108,99,255,0.05) !important;
  margin-bottom: 10px !important;
}}
[data-testid="stFileUploader"] section {{
  padding: 6px 12px !important;
  background: transparent !important;
}}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span {{
  color: {C['purple_dim']} !important;
  font-size: 12px !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] {{
  color: {C['text2']} !important;
  font-size: 12px !important;
}}

/* ── Buttons ── */
/* Secondary / outlined (Demo button) */
.stButton > button[kind="secondary"] {{
  background: transparent !important;
  border: 1.5px solid {C['purple']} !important;
  color: {C['purple']} !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  border-radius: 999px !important;
  padding: 12px 28px !important;
  box-shadow: none !important;
  transition: all .18s ease !important;
  height: 44px !important;
}}
.stButton > button[kind="secondary"]:hover {{
  background: rgba(108,99,255,0.1) !important;
  color: {C['purple_dim']} !important;
  border-color: {C['purple_dim']} !important;
  transform: none !important;
}}

/* Primary / solid (Analyse button) */
.stButton > button[kind="primary"],
.stButton > button:not([kind="secondary"]) {{
  background: {C['purple']} !important;
  border: none !important;
  color: #fff !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  font-family: 'Inter', sans-serif !important;
  border-radius: 999px !important;
  padding: 12px 28px !important;
  box-shadow: 0 4px 18px rgba(108,99,255,.40) !important;
  transition: all .18s ease !important;
  height: 44px !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button:not([kind="secondary"]):hover {{
  background: #7C73FF !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px rgba(108,99,255,.50) !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; }}

/* Download button */
.stDownloadButton > button {{
  background: transparent !important;
  border: 1.5px solid rgba(108,99,255,.45) !important;
  color: {C['purple_dim']} !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  padding: 10px 22px !important;
  box-shadow: none !important;
}}
.stDownloadButton > button:hover {{
  background: rgba(108,99,255,.1) !important;
  transform: none !important;
}}

/* ── Section titles ── */
.sec-title {{
  font-size: 19px; font-weight: 800; color: {C['text']};
  letter-spacing: -.3px; margin: 0 0 4px;
}}
.sec-sub   {{ font-size: 13px; color: {C['text2']}; margin: 0 0 18px; }}
.sec-divider {{ height: 1px; background: {C['divider']}; margin: 32px 0; }}

/* ── Result cards (shared base) ── */
.score-outer, .sub-card, .skills-card,
.quality-card, .mini-card, .chart-card {{
  background: {C['card']};
  border: 1px solid {C['border']};
  border-radius: 14px;
  box-shadow: 0 2px 12px rgba(0,0,0,.25);
}}

/* Score card */
.score-outer {{ padding: 30px 22px 24px; text-align: center; }}
.score-verdict-label {{
  font-size: 11px; font-weight: 600; letter-spacing: .8px;
  text-transform: uppercase; color: {C['text2']}; margin-top: 14px;
}}
.score-verdict {{ font-size: 21px; font-weight: 800; margin-top: 3px; }}
.score-stat-row {{
  display: flex; justify-content: center;
  margin-top: 18px; padding-top: 16px;
  border-top: 1px solid {C['divider']};
}}
.score-stat {{
  flex: 1; text-align: center; padding: 0 6px;
  border-right: 1px solid {C['divider']};
}}
.score-stat:last-child {{ border-right: none; }}
.score-stat-n {{ font-size: 20px; font-weight: 800; }}
.score-stat-l {{ font-size: 10px; color: {C['text2']}; font-weight: 500; margin-top: 2px; }}

/* Sub-score card */
.sub-card {{ padding: 22px 22px 18px; height: 100%; }}
.sub-card-title {{
  font-size: 13.5px; font-weight: 700; color: {C['text']};
  margin: 0 0 16px; padding-bottom: 12px;
  border-bottom: 1px solid {C['divider']};
}}

/* Skills cards */
.skills-card {{ padding: 20px 18px; height: 100%; }}
.skills-card-head {{
  display: flex; align-items: center; gap: 7px;
  padding-bottom: 11px; margin-bottom: 11px;
  border-bottom: 1px solid {C['divider']};
}}
.skills-dot  {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.skills-label {{ font-size: 13px; font-weight: 700; color: {C['text']}; }}
.skills-count {{
  margin-left: auto; font-size: 11.5px; font-weight: 600;
  padding: 2px 8px; border-radius: 999px;
}}
.skill-sem-note {{ font-size: 11px; color: {C['text2']}; margin: 6px 0 4px; }}
.no-skills {{ font-size: 12.5px; color: {C['text2']}; font-style: italic; }}

/* Pills — dark theme */
.pill {{
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  margin: 3px 2px; font-size: 11.5px; font-weight: 500; line-height: 1.5;
}}
.pill-green {{ background: {C['green_bg']}; color: {C['green']};  border: 1px solid {C['green_bdr']}; }}
.pill-blue  {{ background: {C['blue_bg']};  color: {C['blue']};   border: 1px solid {C['blue_bdr']};  }}
.pill-red   {{ background: {C['red_bg']};   color: {C['red']};    border: 1px solid {C['red_bdr']};   }}

/* Quality cards */
.quality-card {{ padding: 20px 20px 16px; height: 100%; }}
.quality-head {{
  font-size: 13.5px; font-weight: 700; color: {C['text']};
  padding-bottom: 11px; margin-bottom: 11px;
  border-bottom: 1px solid {C['divider']};
  display: flex; align-items: center; gap: 7px;
}}
.quality-item {{
  display: flex; gap: 9px; align-items: flex-start; margin-bottom: 9px;
}}
.quality-icon {{
  width: 18px; height: 18px; min-width: 18px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; margin-top: 2px; font-weight: 700;
}}
.quality-text {{ font-size: 12.5px; color: {C['text2']}; line-height: 1.6; }}

/* Mini-cards */
.mini-card {{ padding: 18px 20px; }}
.mini-card-title {{ font-size: 12px; font-weight: 600; color: {C['text2']}; margin-bottom: 5px; }}
.mini-card-num   {{ font-size: 28px; font-weight: 800; line-height: 1; }}
.mini-card-label {{ font-size: 11.5px; color: {C['text2']}; margin-top: 3px; }}
.mini-bar-track  {{
  height: 5px; background: {C['track']}; border-radius: 999px;
  margin-top: 10px; overflow: hidden;
}}
.mini-bar-fill {{ height: 100%; border-radius: 999px; }}

/* Chart card */
.chart-card {{ padding: 22px 22px 14px; }}
.chart-note {{ font-size: 11.5px; color: {C['text2']}; text-align: center; margin-top: 4px; }}

/* Action plan tiers */
.action-tier {{
  border-radius: 12px; padding: 18px 20px;
  border-left: 3px solid transparent; height: 100%;
}}
.tier-critical  {{ background: {C['red_bg']};   border-color: {C['red']};   }}
.tier-important {{ background: {C['amber_bg']}; border-color: {C['amber']}; }}
.tier-nice      {{ background: {C['blue_bg']};  border-color: {C['blue']};  }}
.action-tier-title {{
  font-size: 11.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .6px; margin-bottom: 12px;
}}
.tier-title-critical  {{ color: {C['red']};   }}
.tier-title-important {{ color: {C['amber']}; }}
.tier-title-nice      {{ color: {C['blue']};  }}
.action-item {{
  display: flex; gap: 10px; align-items: flex-start; margin-bottom: 9px;
}}
.action-num {{
  width: 24px; height: 24px; min-width: 24px; border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 800; color: #fff;
  background: {C['purple']};
}}
.action-text  {{ font-size: 12.5px; color: {C['text2']}; line-height: 1.65; margin-top: 3px; }}
.action-empty {{ font-size: 12.5px; color: {C['text3']}; font-style: italic; }}

/* Verdict card */
.verdict-card {{
  background: linear-gradient(135deg, #12102A 0%, #1C1940 100%);
  border: 1px solid rgba(108,99,255,.35); border-radius: 14px;
  padding: 26px 28px;
  box-shadow: 0 4px 24px rgba(108,99,255,.12);
}}
.verdict-icon {{
  width: 38px; height: 38px; border-radius: 9px; background: {C['purple']};
  display: flex; align-items: center; justify-content: center;
  font-size: 19px; margin-bottom: 12px;
}}
.verdict-title {{ font-size: 15px; font-weight: 700; color: {C['text']}; margin-bottom: 9px; }}
.verdict-body  {{ font-size: 13.5px; color: {C['text2']}; line-height: 1.8; }}

/* Footer */
.hm-footer {{
  background: {C['bg']};
  border-top: 1px solid {C['divider']};
  padding: 18px 48px; text-align: center;
}}
.hm-footer p {{ font-size: 12px; color: {C['text3']}; margin: 0; }}
.hm-footer strong {{ color: {C['text2']}; }}

/* Misc */
.stAlert {{ border-radius: 10px !important; }}
.stSpinner > div {{ color: {C['purple']} !important; }}
[data-testid="column"] {{ padding: 0 8px !important; }}
div[data-testid="stVerticalBlock"] > div {{ padding-top: 0 !important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
for key, val in [("step", "input"), ("result", None),
                 ("cv_text", ""), ("jd_text", ""), ("_last_pdf", "")]:
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# Navbar
# ══════════════════════════════════════════════════════════════════════════════
step_num = 1 if st.session_state.step == "input" else 2
s1_cls   = "step-active" if step_num == 1 else "step-done"
s2_cls   = "step-active" if step_num == 2 else "step-idle"
s1_lbl   = "✓" if step_num > 1 else "1"

st.markdown(f"""
<div class="hm-nav">
  <div class="hm-logo">
    <div class="hm-logo-icon">🎯</div>
    HireMatch
    <span class="hm-logo-ai">AI</span>
  </div>
  <div class="hm-nav-steps">
    <span class="step-dot {s1_cls}">{s1_lbl}</span>
    <span>Input</span>
    <div class="step-line"></div>
    <span class="step-dot {s2_cls}">2</span>
    <span>Results</span>
  </div>
  <div class="hm-badge">Semantic AI</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — INPUT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == "input":

    # Hero
    st.markdown("""
    <div class="hm-hero">
      <h1>AI-powered CV analysis that tells you exactly<br>what to fix before you apply</h1>
      <p>Paste your CV and a job description. Get a match score, skill gap analysis,<br>
         ATS rating, keyword insights, and a personalised action plan — in seconds.</p>
      <div class="hm-hero-chips">
        <span class="hm-chip">🧠 Sentence Embeddings</span>
        <span class="hm-chip">🎯 Semantic Matching</span>
        <span class="hm-chip">📊 Keyword Gap Analysis</span>
        <span class="hm-chip">🔥 Skill Heatmap</span>
        <span class="hm-chip">🤖 ATS Compatibility</span>
        <span class="hm-chip">🗺 Action Plan</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hm-wrap">', unsafe_allow_html=True)

    # ── Input columns wrapped in scoping div for card CSS ─────────────────────
    st.markdown('<div class="input-cols">', unsafe_allow_html=True)
    col_cv, col_jd = st.columns(2, gap="medium")

    with col_cv:
        st.markdown("""
        <div class="panel-head">
          <div class="panel-icon">📋</div>
          <div>
            <div class="panel-title">Your CV</div>
            <div class="panel-sub">Paste text or upload a PDF</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload PDF", type=["pdf"], key="pdf_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            uid = uploaded.name + str(uploaded.size)
            if st.session_state["_last_pdf"] != uid:
                extracted = extract_pdf_text(uploaded)
                if extracted:
                    st.session_state.cv_text = extracted
                    st.session_state["_last_pdf"] = uid
                    st.rerun()

        st.text_area(
            "cv", label_visibility="collapsed",
            placeholder="Or paste your CV here — skills, experience, education, projects…",
            height=348, key="cv_text",
        )
        cv_len = len(st.session_state.cv_text)
        cv_wc  = len(st.session_state.cv_text.split())
        cv_col = C["green"] if cv_len >= 500 else C["amber"] if cv_len > 0 else C["text3"]
        st.markdown(
            f'<div class="char-count" style="color:{cv_col};">'
            f'{cv_len:,} chars · {cv_wc:,} words</div>',
            unsafe_allow_html=True,
        )

    with col_jd:
        st.markdown("""
        <div class="panel-head">
          <div class="panel-icon">💼</div>
          <div>
            <div class="panel-title">Job Description</div>
            <div class="panel-sub">Paste the full job posting</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.text_area(
            "jd", label_visibility="collapsed",
            placeholder="Paste the job description here — responsibilities, requirements, nice-to-haves…",
            height=420, key="jd_text",
        )
        jd_len = len(st.session_state.jd_text)
        jd_wc  = len(st.session_state.jd_text.split())
        jd_col = C["green"] if jd_len >= 300 else C["amber"] if jd_len > 0 else C["text3"]
        st.markdown(
            f'<div class="char-count" style="color:{jd_col};">'
            f'{jd_len:,} chars · {jd_wc:,} words</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)  # close .input-cols

    # ── Buttons ───────────────────────────────────────────────────────────────
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    _, c_demo, c_gap, c_run, _ = st.columns([1, 1.1, 0.4, 1.4, 1])

    with c_demo:
        if st.button("Load Demo", use_container_width=True):
            st.session_state.cv_text = SAMPLE_CV
            st.session_state.jd_text = SAMPLE_JD
            st.rerun()

    with c_run:
        if st.button("Analyse My CV  →", use_container_width=True, type="primary"):
            cv_val = st.session_state.cv_text.strip()
            jd_val = st.session_state.jd_text.strip()
            if not cv_val or not jd_val:
                st.warning("Please fill in both your CV and the job description.")
            else:
                with st.spinner("Analysing… (first run downloads the AI model, ~80 MB)"):
                    st.session_state.result = analyze(cv_val, jd_val)
                st.session_state.step = "results"
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # close .hm-wrap


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == "results" and st.session_state.result:
    r     = st.session_state.result
    score = r["score"]

    if score >= 70:
        ring_color = C["green"]; ring_track = C["green_bg"]
        v_lbl = "Strong Match";  v_css = f"color:{C['green']};"
    elif score >= 45:
        ring_color = C["amber"]; ring_track = C["amber_bg"]
        v_lbl = "Partial Match"; v_css = f"color:{C['amber']};"
    else:
        ring_color = C["red"];   ring_track = C["red_bg"]
        v_lbl = "Weak Match";    v_css = f"color:{C['red']};"

    st.markdown('<div class="hm-wrap">', unsafe_allow_html=True)

    # Top action bar
    bar_l, bar_r = st.columns([1, 1])
    with bar_l:
        if st.button("← Back to Editor"):
            st.session_state.step = "input"
            st.rerun()
    with bar_r:
        st.download_button(
            "↓ Export Report",
            data=build_html_report(r).encode("utf-8"),
            file_name="hirematch_report.html",
            mime="text/html",
            use_container_width=True,
        )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── ① Score ───────────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-title">Match Analysis</div>'
                f'<div class="sec-sub">Overall compatibility between your CV and this role</div>',
                unsafe_allow_html=True)

    sc_col, sub_col = st.columns([1, 2.2], gap="medium")

    with sc_col:
        total_m = len(r["matched"]) + len(r["semantic_matched"])
        st.markdown(f"""
        <div class="score-outer">
          {score_ring(score, ring_color, ring_track)}
          <div class="score-verdict-label">Overall Score</div>
          <div class="score-verdict" style="{v_css}">{v_lbl}</div>
          <div class="score-stat-row">
            <div class="score-stat">
              <div class="score-stat-n" style="color:{C['green']};">{total_m}</div>
              <div class="score-stat-l">Matched</div>
            </div>
            <div class="score-stat">
              <div class="score-stat-n" style="color:{C['red']};">{len(r['missing'])}</div>
              <div class="score-stat-l">Missing</div>
            </div>
            <div class="score-stat">
              <div class="score-stat-n" style="color:{C['purple']};">{r['ats_score']:.0f}</div>
              <div class="score-stat-l">ATS</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with sub_col:
        ss = r["sub_scores"]
        def sc(v): return C["green"] if v >= 70 else C["amber"] if v >= 45 else C["red"]
        st.markdown(f"""
        <div class="sub-card">
          <div class="sub-card-title">Detailed Score Breakdown</div>
          {sub_score_bar("Skills Match",          ss["skills_match"],         sc(ss["skills_match"]))}
          {sub_score_bar("Experience Relevance",  ss["experience_relevance"], sc(ss["experience_relevance"]))}
          {sub_score_bar("Keywords Match",        ss["keywords_match"],       sc(ss["keywords_match"]))}
          <div style="margin-top:16px;padding-top:14px;
                      border-top:1px solid {C['divider']};
                      font-size:12.5px;color:{C['text2']};line-height:1.7;">
            {r['verdict'][:230] + '…' if len(r['verdict']) > 230 else r['verdict']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ── ② Skills ──────────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-title">Skills Analysis</div>'
                f'<div class="sec-sub">Exact matches, semantically similar skills, and gaps</div>',
                unsafe_allow_html=True)

    sk1, sk2, sk3 = st.columns(3, gap="medium")

    with sk1:
        ph = "".join(f'<span class="pill pill-green">{s}</span>' for s in r["matched"]) \
             or f'<div class="no-skills">No exact matches found</div>'
        st.markdown(f"""
        <div class="skills-card">
          <div class="skills-card-head">
            <div class="skills-dot" style="background:{C['green']};"></div>
            <span class="skills-label">Matched</span>
            <span class="skills-count"
                  style="background:{C['green_bg']};color:{C['green']};">{len(r['matched'])}</span>
          </div>{ph}
        </div>""", unsafe_allow_html=True)

    with sk2:
        if r["semantic_matched"]:
            ph   = "".join(
                f'<span class="pill pill-blue" title="CV skill: {cv_s}">{jd_s}</span>'
                for jd_s, cv_s in sorted(r["semantic_matched"].items())
            )
            note = f'<div class="skill-sem-note">Hover pill to see matching CV skill</div>'
        else:
            ph   = f'<div class="no-skills">No semantic matches</div>'
            note = ""
        st.markdown(f"""
        <div class="skills-card">
          <div class="skills-card-head">
            <div class="skills-dot" style="background:{C['blue']};"></div>
            <span class="skills-label">Semantic</span>
            <span class="skills-count"
                  style="background:{C['blue_bg']};color:{C['blue']};">{len(r['semantic_matched'])}</span>
          </div>{note}{ph}
        </div>""", unsafe_allow_html=True)

    with sk3:
        freq = r.get("missing_freq", {})
        if r["missing"]:
            ph = "".join(
                f'<span class="pill pill-red">{s}'
                f'{f" (×{freq[s]})" if freq.get(s, 0) > 0 else ""}'
                f'</span>'
                for s in r["missing"]
            )
        else:
            ph = f'<div class="no-skills">No gaps — great coverage!</div>'
        st.markdown(f"""
        <div class="skills-card">
          <div class="skills-card-head">
            <div class="skills-dot" style="background:{C['red']};"></div>
            <span class="skills-label">Missing</span>
            <span class="skills-count"
                  style="background:{C['red_bg']};color:{C['red']};">{len(r['missing'])}</span>
          </div>{ph}
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ── ③ CV Quality ──────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-title">CV Quality Analysis</div>'
                f'<div class="sec-sub">What your CV does well and what\'s holding you back</div>',
                unsafe_allow_html=True)

    qa_l, qa_r = st.columns(2, gap="medium")

    with qa_l:
        items = "".join(
            f'<div class="quality-item">'
            f'<div class="quality-icon" style="background:{C["green_bg"]};color:{C["green"]};">✓</div>'
            f'<div class="quality-text">{html_lib.escape(s)}</div></div>'
            for s in r["strengths"]
        ) or f'<div class="quality-text" style="color:{C["text3"]};font-style:italic;">No clear strengths detected.</div>'
        st.markdown(f"""
        <div class="quality-card">
          <div class="quality-head"><span>💪</span> Strengths</div>
          {items}
        </div>""", unsafe_allow_html=True)

    with qa_r:
        items = "".join(
            f'<div class="quality-item">'
            f'<div class="quality-icon" style="background:{C["red_bg"]};color:{C["red"]};">!</div>'
            f'<div class="quality-text">{html_lib.escape(s)}</div></div>'
            for s in r["weaknesses"]
        ) or f'<div class="quality-text" style="color:{C["text3"]};font-style:italic;">No significant weaknesses.</div>'
        st.markdown(f"""
        <div class="quality-card">
          <div class="quality-head"><span>⚠️</span> Weaknesses</div>
          {items}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Mini metric cards
    def mc(v): return C["green"] if v >= 70 else C["amber"] if v >= 50 else C["red"]

    mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")
    word_count = len(st.session_state.cv_text.split())
    all_m      = len(r["matched"]) + len(r["semantic_matched"])
    total_s    = all_m + len(r["missing"])
    cov_pct    = round(all_m / total_s * 100) if total_s else 0

    for col, title, num, lbl, pct, col_fn in [
        (mc1, "ATS Score",      f"{r['ats_score']:.0f}",
         "Likely to pass" if r["ats_score"] >= 70 else "May struggle" if r["ats_score"] >= 50 else "High ATS risk",
         r["ats_score"], mc),
        (mc2, "CV Readability", f"{r['readability']:.0f}",
         html_lib.escape(r["readability_label"]),
         r["readability"], mc),
        (mc3, "Word Count",     f"{word_count:,}",
         "Good length" if word_count >= 400 else "Expand to 400+ words" if word_count >= 200 else "Too short",
         min(100, word_count / 8), mc),
        (mc4, "Skill Coverage", f"{cov_pct}%",
         f"{all_m} of {total_s} skills covered",
         cov_pct, mc),
    ]:
        with col:
            color = col_fn(pct)
            st.markdown(f"""
            <div class="mini-card">
              <div class="mini-card-title">{title}</div>
              <div class="mini-card-num" style="color:{color};">{num}</div>
              <div class="mini-card-label">{lbl}</div>
              <div class="mini-bar-track">
                <div class="mini-bar-fill"
                     style="width:{min(100,pct):.0f}%;background:{color};"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ATS Compatibility Warnings
    ats_warnings = r.get("ats_warnings", [])
    if ats_warnings:
        warnings_html = "".join(
            f'<div class="quality-item">'
            f'<div class="quality-icon" style="background:{C["amber_bg"]};color:{C["amber"]};">!</div>'
            f'<div class="quality-text">{html_lib.escape(w)}</div></div>'
            for w in ats_warnings
        )
        st.markdown(f"""
        <div style="margin-top:16px;background:{C['card']};border:1px solid rgba(252,211,77,0.22);
                    border-radius:14px;padding:20px 20px 14px;
                    box-shadow:0 2px 12px rgba(0,0,0,.25);">
          <div style="font-size:13.5px;font-weight:700;color:{C['amber']};
                      padding-bottom:11px;margin-bottom:11px;
                      border-bottom:1px solid {C['divider']};
                      display:flex;align-items:center;gap:8px;">
            ⚠️ ATS Compatibility Warnings
          </div>
          {warnings_html}
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ── ④ Keyword Gap Analysis ────────────────────────────────────────────────
    st.markdown(f'<div class="sec-title">Keyword Gap Analysis</div>'
                f'<div class="sec-sub">Top 15 JD keywords — blue outline = JD frequency, orange fill = your CV coverage</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    keyword_chart(r["keyword_density"])
    st.markdown(f'<div class="chart-note">Blue outline = job description · Orange bar = your CV · Sort order = JD importance</div>',
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── ⑤ Skill Similarity Heatmap ────────────────────────────────────────────
    if r.get("skill_heatmap", {}).get("matrix"):
        st.markdown(f'<div class="sec-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-title">Skill Similarity Heatmap</div>'
                    f'<div class="sec-sub">How your CV skills semantically cover the JD\'s requirements (1.00 = perfect match)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        skill_heatmap_chart(r["skill_heatmap"])
        st.markdown(
            f'<div class="chart-note">Bright purple = high semantic similarity · Dark = low overlap · '
            f'Rows = JD required skills · Columns = your CV skills</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f'<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ── ⑥ Action Plan ─────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-title">Personalised Action Plan</div>'
                f'<div class="sec-sub">Specific, prioritised steps based on your actual CV and this JD</div>',
                unsafe_allow_html=True)

    ap = r["action_plan"]
    apc1, apc2, apc3 = st.columns(3, gap="medium")

    def render_tier(items, tier_css, title_css, label, icon):
        if items:
            body = "".join(
                f'<div class="action-item"><div class="action-num">{i+1}</div>'
                f'<div class="action-text">{html_lib.escape(item)}</div></div>'
                for i, item in enumerate(items)
            )
        else:
            body = f'<div class="action-empty">No {label.split("—")[1].strip().lower()} actions needed.</div>'
        return (f'<div class="action-tier {tier_css}">'
                f'<div class="action-tier-title {title_css}">{icon} {html_lib.escape(label)}</div>'
                f'{body}</div>')

    with apc1:
        st.markdown(render_tier(ap["critical"],    "tier-critical",  "tier-title-critical",  "P1 — Critical",     "🔴"), unsafe_allow_html=True)
    with apc2:
        st.markdown(render_tier(ap["important"],   "tier-important", "tier-title-important", "P2 — Important",    "🟡"), unsafe_allow_html=True)
    with apc3:
        st.markdown(render_tier(ap["nice_to_have"],"tier-nice",      "tier-title-nice",      "P3 — Nice to Have", "🔵"), unsafe_allow_html=True)

    st.markdown(f'<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ── ⑦ Verdict ─────────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-title">Final Verdict</div>'
                f'<div class="sec-sub">Your personalised career-coach summary</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="verdict-card">
      <div class="verdict-icon">🎯</div>
      <div class="verdict-title">Career Coach Assessment</div>
      <div class="verdict-body">{html_lib.escape(r['verdict'])}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    _, dl_col, _ = st.columns([1.5, 1, 1.5])
    with dl_col:
        st.download_button(
            "↓ Download Full Report",
            data=build_html_report(r).encode("utf-8"),
            file_name="hirematch_report.html",
            mime="text/html",
            use_container_width=True,
            key="export_bottom",
        )

    st.markdown("</div>", unsafe_allow_html=True)  # close .hm-wrap


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hm-footer">
  <p>
    <strong>HireMatch AI</strong> &nbsp;·&nbsp;
    Powered by <strong>all-MiniLM-L6-v2</strong> sentence embeddings
    &nbsp;·&nbsp; Semantic threshold: 0.70 cosine similarity
    &nbsp;·&nbsp; Results are indicative, not a hiring guarantee
  </p>
</div>
""", unsafe_allow_html=True)
