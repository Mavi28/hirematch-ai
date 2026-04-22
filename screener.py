"""
screener.py — HireMatch AI: comprehensive CV analysis backend.

Analysis layers:
  1. Sentence-transformer embeddings (all-MiniLM-L6-v2) → experience relevance proxy
  2. Exact keyword matching → skills gap
  3. Semantic keyword matching at skill level (cosine ≥ 0.70)
  4. Keyword density (CountVectorizer) → top-20 JD terms vs CV coverage
  5. Sub-scores: skills match % (frequency-weighted), experience relevance %, keywords density %
  6. Overall score = Skills 40% + Experience 30% + Keywords 30%
  7. ATS score estimate (heuristic, 0–100) with specific compatibility warnings
  8. Readability score (simplified Flesch Reading Ease)
  9. Skill similarity heatmap (pairwise cosine between top-8 JD & CV skills)
  10. Strengths + weaknesses (data-driven, specific to actual CV/JD)
  11. Prioritised action plan (critical / important / nice-to-have)
  12. Career-coach verdict paragraph
"""

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ─────────────────────────────────────────────────────────────────────
SEMANTIC_THRESHOLD = 0.70

_model = None


def _get_model():
    """Lazy-load SentenceTransformer once; reuse across Streamlit reruns."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ── Skill keyword vocabulary ────────────────────────────────────────────────────
SKILL_KEYWORDS = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
    # Frontend
    "html", "css", "react", "angular", "vue", "next.js", "nuxt", "svelte",
    "bootstrap", "tailwind", "jquery", "webpack", "vite",
    # Backend / APIs
    "node.js", "django", "flask", "fastapi", "spring", "express", "graphql",
    "rest", "api", "microservices", "grpc",
    # Data / ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "opencv", "hugging face", "llm",
    "feature engineering", "model deployment", "a/b testing",
    # MLOps
    "mlops", "mlflow", "kubeflow", "airflow", "spark", "hadoop",
    "data engineering", "data pipelines", "experiment tracking",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle", "cassandra", "dynamodb", "firebase",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd", "linux", "bash", "git",
    # Soft skills
    "communication", "teamwork", "leadership", "problem solving",
    "critical thinking", "agile", "scrum", "project management", "mentoring",
    # General tech
    "data analysis", "excel", "tableau", "power bi", "jira", "confluence",
    "prompt engineering", "fine-tuning", "distributed training",
]


# ── Text utilities ──────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\+#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _count_keyword(text: str, keyword: str) -> int:
    """Count word-boundary occurrences of keyword in text (case-insensitive)."""
    return len(re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", text.lower()))


def _extract_skills(text: str) -> set[str]:
    cleaned = _clean_text(text)
    return {
        skill for skill in SKILL_KEYWORDS
        if re.search(r"\b" + re.escape(skill) + r"\b", cleaned)
    }


# ── Semantic skill matching ─────────────────────────────────────────────────────

def _encode_skills(skills: list[str]) -> np.ndarray:
    return _get_model().encode(skills, convert_to_numpy=True, normalize_embeddings=True)


def _semantic_skill_match(
    cv_skills: set[str],
    jd_gap: set[str],
) -> tuple[dict[str, str], set[str]]:
    """Map each un-matched JD skill to the closest CV skill (if ≥ threshold)."""
    if not jd_gap or not cv_skills:
        return {}, jd_gap

    cv_list, jd_list = sorted(cv_skills), sorted(jd_gap)
    sim = cosine_similarity(_encode_skills(jd_list), _encode_skills(cv_list))

    matched, missing = {}, set()
    for i, jd_skill in enumerate(jd_list):
        best = int(np.argmax(sim[i]))
        if float(sim[i, best]) >= SEMANTIC_THRESHOLD:
            matched[jd_skill] = cv_list[best]
        else:
            missing.add(jd_skill)
    return matched, missing


# ── Keyword density ─────────────────────────────────────────────────────────────

def _keyword_density(cv_text: str, jd_text: str, top_n: int = 20) -> list[dict]:
    """Top N meaningful JD words with their frequency in JD vs CV."""
    try:
        jd_clean = _clean_text(jd_text)
        cv_clean = _clean_text(cv_text)

        vec = CountVectorizer(stop_words="english", ngram_range=(1, 1), min_df=1)
        vec.fit([jd_clean])
        features = vec.get_feature_names_out()

        jd_counts = vec.transform([jd_clean]).toarray()[0]
        cv_vec    = CountVectorizer(vocabulary=vec.vocabulary_, ngram_range=(1, 1))
        cv_counts = cv_vec.transform([cv_clean]).toarray()[0]

        rows = [
            {"keyword": features[i], "jd_count": int(jd_counts[i]), "cv_count": int(cv_counts[i])}
            for i in range(len(features))
            if len(features[i]) >= 4 and jd_counts[i] >= 2
        ]
        rows.sort(key=lambda x: -x["jd_count"])
        return rows[:top_n]
    except Exception:
        return []


# ── Experience relevance ────────────────────────────────────────────────────────

def _extract_years(text: str) -> int | None:
    """Extract max years of experience mentioned in text."""
    matches = re.findall(r"(\d+)\+?\s*(?:to\s*\d+\s*)?year", text.lower())
    return max(int(m) for m in matches) if matches else None


def _compute_experience_score(cv_text: str, jd_text: str) -> float:
    """Score experience relevance: seniority match, years, title keywords, industry."""
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    score    = 0.0

    # Seniority level match (0–30 pts)
    levels = {
        "intern": 0, "entry": 1, "junior": 1, "associate": 1,
        "mid": 2, "intermediate": 2, "senior": 3, "lead": 4,
        "principal": 4, "staff": 4, "manager": 5, "director": 6,
        "vp": 7, "head": 5,
    }
    jd_rank = next((r for k, r in levels.items() if k in jd_lower), None)
    cv_rank = next((r for k, r in levels.items() if k in cv_lower), None)
    if jd_rank is not None and cv_rank is not None:
        diff    = abs(jd_rank - cv_rank)
        score  += 30 if diff == 0 else 20 if diff == 1 else 10 if diff <= 2 else 0
    else:
        score += 15  # neutral

    # Years of experience match (0–30 pts)
    jd_years = _extract_years(jd_text)
    cv_years = _extract_years(cv_text)
    if jd_years and cv_years:
        score += 30 if cv_years >= jd_years else 20 if cv_years >= jd_years * 0.7 else 10
    elif cv_years:
        score += 20  # CV mentions years, JD doesn't specify a minimum
    else:
        score += 15  # neutral

    # Job title / role keyword overlap in first ~300 chars (0–20 pts)
    jd_words = set(re.findall(r"\b[a-z]{4,}\b", jd_lower[:300]))
    cv_words = set(re.findall(r"\b[a-z]{4,}\b", cv_lower[:300]))
    score   += min(20, len(jd_words & cv_words) * 2)

    # Industry keyword match (0–20 pts)
    industry_kws = [
        "fintech", "healthcare", "ecommerce", "saas", "enterprise", "startup",
        "b2b", "b2c", "finance", "banking", "retail", "logistics", "education",
        "security", "gaming", "media", "telecommunications", "insurance",
    ]
    jd_ind = {k for k in industry_kws if k in jd_lower}
    cv_ind = {k for k in industry_kws if k in cv_lower}
    score  += min(20, (len(jd_ind & cv_ind) / len(jd_ind) * 20) if jd_ind else 10)

    return round(min(100.0, score), 1)


# ── Sub-scores ──────────────────────────────────────────────────────────────────

def _compute_sub_scores(
    cv_skills: set[str],
    jd_skills: set[str],
    semantic_matched: dict,
    cv_text: str,
    jd_text: str,
    kw_data: list[dict],
) -> dict:
    # 1. Skills Match: each JD skill weighted by its JD frequency
    if jd_skills:
        total_w = covered_w = 0
        for skill in jd_skills:
            w        = max(1, _count_keyword(jd_text, skill))
            total_w += w
            if skill in cv_skills or skill in semantic_matched:
                covered_w += w
        skills_match = min(100.0, covered_w / total_w * 100) if total_w else 0.0
    else:
        skills_match = 50.0

    # 2. Experience Relevance: seniority + years + title/industry heuristics
    experience_relevance = _compute_experience_score(cv_text, jd_text)

    # 3. Keywords Density: top-20 JD keyword coverage (kw_data already top-20)
    if kw_data:
        total_jd   = sum(k["jd_count"] for k in kw_data)
        covered_kw = sum(min(k["cv_count"], k["jd_count"]) for k in kw_data)
        kw_match   = min(100.0, covered_kw / total_jd * 100) if total_jd else 50.0
    else:
        kw_match = 50.0

    # Weighted overall — Skills 40% + Experience 30% + Keywords 30%
    overall = round(0.4 * skills_match + 0.3 * experience_relevance + 0.3 * kw_match, 1)

    return {
        "skills_match":         round(skills_match, 1),
        "experience_relevance": round(experience_relevance, 1),
        "keywords_match":       round(kw_match, 1),
        "overall":              overall,
    }


# ── ATS score ───────────────────────────────────────────────────────────────────

def _compute_ats_score(
    cv_text: str,
    jd_skills: set[str],
    matched: list,
    semantic_matched: dict,
) -> tuple[float, list[str]]:
    score    = 0.0
    warnings = []
    cv_lower = cv_text.lower()
    lines    = cv_text.split("\n")

    # Skill coverage (30 pts)
    if jd_skills:
        score += min(30.0, (len(matched) + len(semantic_matched)) / len(jd_skills) * 30)
    else:
        score += 15.0

    # CV word count (20 pts)
    words  = len(cv_text.split())
    score += 20.0 if words >= 400 else 13.0 if words >= 250 else 7.0 if words >= 150 else 0.0

    # Table detection (−10 pts)
    table_lines = [l for l in lines if l.count("|") >= 2]
    if len(table_lines) >= 3:
        score = max(0, score - 10)
        warnings.append(
            "Tables detected — ATS systems cannot reliably parse table content. "
            "Convert to plain-text bullet lists."
        )

    # Special character overuse (−5 pts)
    special_count = len(re.findall(r"[%$€£¥@#&*]", cv_text))
    if special_count > 15:
        score = max(0, score - 5)
        warnings.append(
            f"High special-character count ({special_count}) — decorative symbols "
            f"can confuse ATS parsers. Keep only functional ones where needed."
        )

    # Very short bullets (warning only)
    bullet_lines  = [l.strip() for l in lines if re.match(r"^[-•*·]\s+", l.strip())]
    short_bullets = [l for l in bullet_lines if len(l.split()) < 6]
    if len(short_bullets) >= 2:
        warnings.append(
            f"{len(short_bullets)} bullet point(s) under 5 words — expand them with "
            f"context, tools used, and measurable outcomes."
        )

    # Quantified achievements (+10 or −5 pts)
    has_numbers = bool(re.search(
        r"\d+\s*(%|percent|\bx\b|times|users|customers|projects|million|\bk\b)", cv_text, re.I
    ))
    if has_numbers:
        score += 10.0
    else:
        score = max(0, score - 5)
        warnings.append(
            "No quantified achievements — ATS and recruiters rank CVs with metrics higher. "
            "Add numbers: '40% faster', '10K users', '3 engineers led'."
        )

    # Dedicated skills section (15 pts)
    if any(kw in cv_lower for kw in ["skills", "technologies", "expertise", "competencies", "technical"]):
        score += 15.0
    else:
        warnings.append(
            "No dedicated Skills section — add one so ATS can reliably extract "
            "your technical skills and map them to the JD."
        )

    # Action verbs (10 pts)
    verbs = [
        "developed", "built", "designed", "led", "managed", "implemented",
        "created", "delivered", "improved", "reduced", "increased", "achieved",
        "architected", "launched", "optimised", "optimized", "deployed",
    ]
    score += min(10.0, sum(1 for v in verbs if v in cv_lower) * 1.5)

    # Contact details / professional links (5 pts)
    if any(kw in cv_lower for kw in ["@", "linkedin", "github", "gitlab", "phone"]):
        score += 5.0
    else:
        warnings.append(
            "No contact details or professional links — include email, "
            "LinkedIn, or GitHub at the top of your CV."
        )

    # Header / footer artefacts (−3 pts)
    if re.search(r"page\s+\d|confidential|curriculum vitae|\s-\s\d\s+-\s", cv_lower):
        score = max(0, score - 3)
        warnings.append(
            "Header/footer artefacts detected — remove page numbers and repeated "
            "labels that confuse ATS parsers."
        )

    return round(min(100.0, score), 1), warnings[:5]


# ── Skill similarity heatmap ────────────────────────────────────────────────────

def _compute_skill_heatmap(cv_skills: set[str], jd_skills: set[str], top_n: int = 8) -> dict:
    """Pairwise cosine similarity between top JD skills and top CV skills."""
    if len(cv_skills) < 2 or len(jd_skills) < 2:
        return {"jd_skills": [], "cv_skills": [], "matrix": []}

    jd_list = sorted(jd_skills)[:top_n]
    cv_list = sorted(cv_skills)[:top_n]

    jd_embs = _encode_skills(jd_list)
    cv_embs = _encode_skills(cv_list)
    sim     = cosine_similarity(jd_embs, cv_embs)

    return {
        "jd_skills": jd_list,
        "cv_skills": cv_list,
        "matrix":    [[round(float(v), 3) for v in row] for row in sim],
    }


# ── Readability ─────────────────────────────────────────────────────────────────

def _compute_readability(text: str) -> tuple[float, str]:
    """Simplified Flesch Reading Ease → (score, label)."""
    sentences  = max(1, len(re.findall(r"[.!?]+", text)))
    words_list = text.split()
    words      = max(1, len(words_list))

    syllables = 0
    for word in words_list:
        w = re.sub(r"[^a-zA-Z]", "", word).lower()
        if not w:
            continue
        cnt = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and cnt > 1:
            cnt -= 1
        syllables += max(1, cnt)

    fre = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    fre = round(max(0.0, min(100.0, fre)), 1)

    if   fre >= 70: label = "Easy to read"
    elif fre >= 50: label = "Moderate"
    elif fre >= 30: label = "Somewhat complex"
    else:           label = "Dense / complex"

    return fre, label


# ── Strengths ───────────────────────────────────────────────────────────────────

def _generate_strengths(
    matched: list,
    semantic_matched: dict,
    cv_text: str,
    kw_data: list[dict],
    sub_scores: dict,
) -> list[str]:
    strengths = []
    cv_lower  = cv_text.lower()

    if sub_scores["skills_match"] >= 55:
        strengths.append(
            f"Strong technical alignment — {len(matched)} required skill(s) appear directly "
            f"in your CV, covering {sub_scores['skills_match']:.0f}% of the role's skill requirements."
        )

    if len(semantic_matched) >= 2:
        ex_jd, ex_cv = next(iter(semantic_matched.items()))
        strengths.append(
            f"Broad transferable expertise — your CV covers {len(semantic_matched)} JD requirement(s) "
            f"semantically (e.g. your '{ex_cv}' experience addresses the JD's '{ex_jd}' requirement)."
        )

    if re.search(r"\d+\s*(%|percent|\bx\b|times|users|customers|projects|million|\bk\b)", cv_text, re.I):
        strengths.append(
            "Quantified impact — your CV includes measurable results, which rank significantly "
            "higher with both ATS systems and human recruiters."
        )

    action_verbs = ["developed", "built", "designed", "led", "managed", "implemented",
                    "launched", "deployed", "architected", "delivered", "improved"]
    found = [v for v in action_verbs if v in cv_lower]
    if len(found) >= 3:
        strengths.append(
            f"Strong ownership language — verbs like '{found[0]}', '{found[1]}', and "
            f"'{found[2]}' signal hands-on contribution and initiative."
        )

    top_covered = [k for k in kw_data[:3] if k["cv_count"] > 0]
    if top_covered:
        kw = top_covered[0]
        strengths.append(
            f"Key term coverage — '{kw['keyword']}' is a high-frequency JD term (×{kw['jd_count']}) "
            f"and appears {kw['cv_count']}× in your CV, improving ATS scoring."
        )

    return strengths[:4]


# ── Weaknesses ──────────────────────────────────────────────────────────────────

def _generate_weaknesses(
    missing: list,
    missing_freq: dict,
    cv_text: str,
    jd_text: str,
    kw_data: list[dict],
    sub_scores: dict,
) -> list[str]:
    weaknesses = []

    for skill in missing[:2]:
        jd_cnt = missing_freq.get(skill, _count_keyword(jd_text, skill))
        weaknesses.append(
            f"Missing '{skill}' — appears {jd_cnt}× in the job description "
            f"but is completely absent from your CV."
        )

    zero_kw = [k for k in kw_data if k["cv_count"] == 0 and k["jd_count"] >= 3]
    if zero_kw:
        kw = zero_kw[0]
        weaknesses.append(
            f"Keyword gap — '{kw['keyword']}' appears {kw['jd_count']}× in the JD "
            f"but 0× in your CV. This is likely an ATS filter term."
        )

    if not re.search(r"\d+\s*(%|percent|\bx\b|times|users|customers|projects)", cv_text, re.I):
        weaknesses.append(
            "No quantified achievements — every bullet is a responsibility, not a result. "
            "Add numbers: '40% faster', '10K users', '3 engineers led'."
        )

    words = len(cv_text.split())
    if words < 250:
        weaknesses.append(
            f"CV is too sparse ({words} words) — ATS systems score thin CVs lower. "
            f"Expand to at least 400 words."
        )

    low = [k for k in kw_data if 0 < k["cv_count"] < k["jd_count"] // 2 and k["jd_count"] >= 4]
    if low:
        kw = low[0]
        weaknesses.append(
            f"Keyword under-use — '{kw['keyword']}' appears {kw['jd_count']}× in the JD "
            f"but only {kw['cv_count']}× in your CV."
        )

    return weaknesses[:4]


# ── Action plan ─────────────────────────────────────────────────────────────────

def _generate_action_plan(
    missing: list,
    missing_freq: dict,
    semantic_matched: dict,
    cv_text: str,
    jd_text: str,
    kw_data: list[dict],
) -> dict:
    critical, important, nice = [], [], []
    cv_lower  = cv_text.lower()
    cv_skills = _extract_skills(cv_text)

    # Critical: missing skills appearing 2+ times in JD
    for skill in missing:
        cnt = missing_freq.get(skill, _count_keyword(jd_text, skill))
        if cnt >= 2:
            critical.append(
                f"Add '{skill}' to your CV — appears {cnt}× in this job description "
                f"but 0 times in your CV. This is almost certainly an ATS filter."
            )

    # Critical: zero-coverage high-frequency JD keywords
    for kw in kw_data[:6]:
        if kw["cv_count"] == 0 and kw["jd_count"] >= 4:
            if not any(kw["keyword"] in c for c in critical):
                critical.append(
                    f"Include '{kw['keyword']}' in your CV — appears {kw['jd_count']}× "
                    f"in the JD but 0× in your CV. High-frequency ATS filter term."
                )

    # Important: semantic matches → suggest exact JD wording
    for jd_skill, cv_skill in list(semantic_matched.items())[:3]:
        if jd_skill != cv_skill:
            important.append(
                f"Good: your '{cv_skill}' covers '{jd_skill}' semantically — but "
                f"consider adding the exact term '{jd_skill}' to your CV for ATS compatibility."
            )

    # Important: keyword terminology mismatch (CV uses related but different term)
    for kw in kw_data:
        if kw["cv_count"] == 0 and kw["jd_count"] >= 3 and len(important) < 2:
            for cv_s in cv_skills:
                if (any(part in kw["keyword"] for part in cv_s.split()) or
                        any(part in cv_s for part in kw["keyword"].split())):
                    important.append(
                        f"Your CV uses '{cv_s}' but this JD uses '{kw['keyword']}' "
                        f"({kw['jd_count']}× in JD) — they may mean the same thing but "
                        f"ATS won't match them. Add '{kw['keyword']}' explicitly."
                    )
                    break

    # Important: quantification
    if not re.search(r"\d+\s*(%|percent|\bx\b|times|users|customers|projects)", cv_text, re.I):
        important.append(
            "Add 3+ quantified results to your experience bullets — e.g. "
            "'Reduced build time by 45%', 'Scaled service to 100K users', "
            "'Led a team of 4'. Metrics dramatically increase interview callbacks."
        )

    # Important: under-used keywords
    for kw in kw_data:
        if 0 < kw["cv_count"] < kw["jd_count"] // 2 and kw["jd_count"] >= 4:
            important.append(
                f"Use '{kw['keyword']}' more prominently — appears {kw['jd_count']}× "
                f"in the JD but only {kw['cv_count']}× in your CV. Mirror the JD's language."
            )
            break

    # Nice to have: professional summary
    if not any(kw in cv_lower for kw in ["summary", "objective", "profile", "about me"]):
        nice.append(
            "Add a 2–3 sentence professional summary at the top of your CV that "
            "names the role you're applying for and your 2 strongest matching skills."
        )

    # Nice to have: portfolio / links
    if not any(kw in cv_lower for kw in ["github", "portfolio", "linkedin", "gitlab"]):
        nice.append(
            "Link to a GitHub profile or project portfolio — technical hiring managers "
            "frequently check these before deciding to interview."
        )

    # Nice to have: length
    words = len(cv_text.split())
    if words < 350:
        nice.append(
            f"Expand your CV from {words} to 400+ words by elaborating on project "
            f"outcomes and specific technologies used in each role."
        )
    elif words > 900:
        nice.append(
            f"Your CV is long ({words} words). Consider trimming roles older than "
            f"10 years to a single line — recruiters focus on recent work."
        )

    return {"critical": critical[:3], "important": important[:3], "nice_to_have": nice[:3]}


# ── Verdict ─────────────────────────────────────────────────────────────────────

def _generate_verdict(
    score: float,
    sub_scores: dict,
    matched: list,
    missing: list,
    semantic_matched: dict,
    cv_text: str,
    ats_score: float,
) -> str:
    total = len(matched) + len(semantic_matched)

    if score >= 70:
        opening = f"Your CV is a strong fit for this role, scoring {score:.1f}% overall."
        middle  = (
            f" You cover {total} of the required skills directly, your keyword alignment "
            f"sits at {sub_scores['keywords_match']:.0f}%, and your estimated ATS score "
            f"is {ats_score:.0f}/100."
        )
        closing = (
            (f" Before submitting, address the {len(missing)} missing skill(s) — "
             f"particularly '{missing[0]}' — to maximise your ATS ranking. "
             f"With those additions, this CV is ready to submit.")
            if missing else
            " There are no critical skill gaps. Apply with confidence."
        )
    elif score >= 45:
        opening = f"Your CV is a partial match ({score:.1f}%) — clear potential but targeted gaps to fix first."
        middle  = (
            f" You match {total} skill(s), but {len(missing)} required skill(s) are missing "
            f"and keyword alignment is {sub_scores['keywords_match']:.0f}%."
        )
        closing = (
            " Spend 1–2 hours incorporating missing terms and mirroring the JD's phrasing "
            "in your experience bullets. This targeted effort can lift your score by 15–25 "
            "points and significantly improve your ATS pass rate."
        )
    else:
        opening = f"Your CV has a low match score ({score:.1f}%) for this specific role."
        middle  = (
            f" Only {total} skill(s) overlap with requirements, {len(missing)} key skills "
            f"are entirely absent, and keyword alignment is just {sub_scores['keywords_match']:.0f}%."
        )
        closing = (
            f" A targeted rewrite is recommended before applying. Prioritise: "
            f"(1) adding the {min(3, len(missing))} most-mentioned missing skills, "
            f"(2) mirroring the JD's exact terminology, and (3) restructuring experience "
            f"bullets around outcomes rather than duties."
        )

    return opening + middle + closing


# ── Public API ──────────────────────────────────────────────────────────────────

def analyze(cv_text: str, jd_text: str) -> dict:
    """
    Full analysis of a CV against a job description.

    Returns a dict with:
        score, embedding_score, matched, semantic_matched, missing, missing_freq,
        sub_scores, ats_score, ats_warnings, readability, readability_label,
        keyword_density, skill_heatmap, strengths, weaknesses, action_plan, verdict
    """
    cv_clean = _clean_text(cv_text)
    jd_clean = _clean_text(jd_text)

    if not cv_clean or not jd_clean:
        return {
            "score": 0.0, "embedding_score": 0.0,
            "matched": [], "semantic_matched": {}, "missing": [], "missing_freq": {},
            "sub_scores": {"skills_match": 0, "experience_relevance": 0,
                           "keywords_match": 0, "overall": 0},
            "ats_score": 0.0, "ats_warnings": [],
            "readability": 0.0, "readability_label": "N/A",
            "keyword_density": [],
            "skill_heatmap": {"jd_skills": [], "cv_skills": [], "matrix": []},
            "strengths": [], "weaknesses": [],
            "action_plan": {"critical": [], "important": [], "nice_to_have": []},
            "verdict": "Please provide both a CV and a job description.",
        }

    # 1. Embedding similarity (kept as experience relevance baseline)
    model           = _get_model()
    cv_emb, jd_emb  = model.encode([cv_clean, jd_clean],
                                    convert_to_numpy=True, normalize_embeddings=True)
    embedding_score = round(float(np.dot(cv_emb, jd_emb)) * 100, 1)

    # 2. Exact keyword matching
    cv_skills = _extract_skills(cv_text)
    jd_skills = _extract_skills(jd_text)
    exact     = cv_skills & jd_skills
    gap       = jd_skills - cv_skills

    # 3. Semantic skill matching on the gap
    semantic_matched, truly_missing = _semantic_skill_match(cv_skills, gap)

    # 4. Keyword density — top-20 for scoring, top-15 for display
    kw_data_20 = _keyword_density(cv_text, jd_text, top_n=20)
    kw_data    = kw_data_20[:15]

    # 5. Sub-scores + weighted overall (Skills 40% + Experience 30% + Keywords 30%)
    sub_scores = _compute_sub_scores(
        cv_skills, jd_skills, semantic_matched, cv_text, jd_text, kw_data_20
    )
    score = sub_scores["overall"]

    # 6. ATS + readability
    ats_score, ats_warnings   = _compute_ats_score(cv_text, jd_skills, sorted(exact), semantic_matched)
    readability, rd_label     = _compute_readability(cv_text)

    # 7. Sort missing skills by JD frequency (most important first)
    missing_with_freq = sorted(
        [{"skill": s, "jd_count": _count_keyword(jd_text, s)} for s in truly_missing],
        key=lambda x: -x["jd_count"],
    )
    missing_list = [m["skill"] for m in missing_with_freq]
    missing_freq = {m["skill"]: m["jd_count"] for m in missing_with_freq}

    # 8. Skill similarity heatmap
    skill_heatmap = _compute_skill_heatmap(cv_skills, jd_skills)

    # 9. Qualitative analysis
    matched_list = sorted(exact)

    strengths   = _generate_strengths(matched_list, semantic_matched, cv_text, kw_data, sub_scores)
    weaknesses  = _generate_weaknesses(missing_list, missing_freq, cv_text, jd_text, kw_data, sub_scores)
    action_plan = _generate_action_plan(
        missing_list, missing_freq, semantic_matched, cv_text, jd_text, kw_data
    )
    verdict = _generate_verdict(
        score, sub_scores, matched_list, missing_list, semantic_matched, cv_text, ats_score
    )

    return {
        "score":             score,
        "embedding_score":   embedding_score,
        "matched":           matched_list,
        "semantic_matched":  semantic_matched,
        "missing":           missing_list,
        "missing_freq":      missing_freq,
        "sub_scores":        sub_scores,
        "ats_score":         ats_score,
        "ats_warnings":      ats_warnings,
        "readability":       readability,
        "readability_label": rd_label,
        "keyword_density":   kw_data,
        "skill_heatmap":     skill_heatmap,
        "strengths":         strengths,
        "weaknesses":        weaknesses,
        "action_plan":       action_plan,
        "verdict":           verdict,
    }
