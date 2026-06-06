import streamlit as st
import pdfplumber
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from skills import skill_score
from colleges import TIER_COLLEGES

st.set_page_config(page_title="BlackBoxCV", layout="wide")

st.title("BlackBoxCV")
st.subheader("Opening the black box of resume screening")

st.write("""
BlackBoxCV helps students understand how automated screening systems
evaluate resumes — beyond skills.
""")

job_description = st.text_area(
    "Paste the Job Description",
    height=150
)

uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type="pdf"
)


def extract_text(file):
    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def mask_colleges(text):
    masked = text.lower()

    for college in TIER_COLLEGES:
        masked = re.sub(college, "[INSTITUTION]", masked)

    return masked


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


def semantic_score(resume_text, job_text):
    embeddings = model.encode(
        [resume_text, job_text],
        convert_to_tensor=True
    )

    return util.cos_sim(
        embeddings[0],
        embeddings[1]
    ).item()


def final_score(resume_text, job_text):
    skill, matched = skill_score(resume_text)

    semantic = semantic_score(
        resume_text,
        job_text
    )

    total = (skill * 0.6) + (semantic * 10)

    return (
        round(total, 2),
        matched,
        round(semantic, 3)
    )


# -----------------------------
# MAIN ANALYSIS SECTION
# -----------------------------

if uploaded_file and job_description:

    try:

        resume_text = extract_text(uploaded_file)

        if not resume_text.strip():
            st.error(
                "Could not extract text from the uploaded PDF."
            )
            st.stop()

        masked_text = mask_colleges(resume_text)

        with st.expander("View Masked Changes"):

            st.write("Original Resume Snippet")
            st.text(resume_text[:500])

            st.write("Masked Resume Snippet")
            st.text(masked_text[:500])

        # Original Resume Evaluation
        score_original, skills_orig, sem_orig = final_score(
            resume_text,
            job_description
        )

        # Masked Resume Evaluation
        score_masked, skills_mask, sem_mask = final_score(
            masked_text,
            job_description
        )

        bias_delta = round(
            score_original - score_masked,
            2
        )

        impact_percent = (
            round(
                (bias_delta / score_original) * 100,
                2
            )
            if score_original != 0
            else 0
        )

        # -----------------------------
        # RESULTS TABLE
        # -----------------------------

        st.subheader("Resume Evaluation Results")

        df = pd.DataFrame({
            "Version": [
                "Original Resume",
                "Masked Institution"
            ],
            "Final Score": [
                score_original,
                score_masked
            ],
            "Semantic Match": [
                sem_orig,
                sem_mask
            ]
        })

        st.dataframe(
            df,
            use_container_width=True
        )

        # -----------------------------
        # INSIGHTS
        # -----------------------------

        st.subheader("What the System Infers")

        if bias_delta > 0:

            st.warning(
                f"""
Your resume gains **{bias_delta} points ({impact_percent}%)**
due to institutional signals.

This indicates that part of the evaluation may depend on
non-skill factors rather than qualifications alone.
"""
            )

        else:

            st.success(
                """
Your resume evaluation appears primarily skill-driven.
Institutional masking did not significantly affect results.
"""
            )

        # -----------------------------
        # DISCLAIMER
        # -----------------------------

        st.info(
            """
Disclaimer:

BlackBoxCV does not claim to replicate real ATS systems exactly.

It demonstrates how proxy signals can influence automated resume evaluation and helps students understand potential screening biases.

The results should be interpreted as educational insights rather than hiring predictions.
"""
        )

    except Exception as e:

        st.error(
            f"An error occurred while processing the resume: {str(e)}"
        )

else:

    st.info(
        """
To begin:

1. Paste a Job Description
2. Upload a Resume PDF

BlackBoxCV will compare the original and institution-masked versions of your resume.
"""
    )