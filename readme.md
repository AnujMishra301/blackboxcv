BlackBoxCV

BlackBoxCV is an NLP-based resume analysis tool that demonstrates how automated resume screening systems evaluate candidates beyond just skills. The project explores how factors like institutional signals can influence automated evaluations by comparing resume scores before and after masking such information.

The goal of the project is to help students understand how AI-assisted hiring tools may interpret resumes and to increase transparency in automated screening processes.

Features

Resume parsing from PDF files

Skill extraction from resume text

Semantic similarity matching between resume and job description

Institutional bias experiment using masked resume data

Interactive web interface for resume evaluation

How It Works

The system evaluates resumes using two primary components:

1. Skill Matching

The system checks the resume text for relevant technical skills and calculates a skill score based on detected technologies.

2. Semantic Similarity

A transformer-based language model compares the resume content and the job description to determine how closely they match semantically.

3. Bias Demonstration

To demonstrate potential institutional bias in automated systems, the application masks well-known university names in the resume and recalculates the score. The difference between the two scores indicates how institutional signals may influence automated evaluation.

System Workflow
Upload Resume (PDF)
        ↓
Extract Resume Text
        ↓
Mask Institutional Names
        ↓
Compute Skill Score
        ↓
Compute Semantic Similarity
        ↓
Generate Resume Evaluation Results
Technologies Used
Programming Language

Python

Libraries and Frameworks

Streamlit

Sentence Transformers

PDFPlumber

Pandas

NumPy

Machine Learning

Transformer-based embeddings for semantic similarity

Project Structure
BlackBoxCV
│
├── app.py              # Main Streamlit application
├── skills.py           # Skill extraction logic
├── colleges.py         # Institutional masking dataset
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
Installation

Clone the repository:

git clone https://github.com/yourusername/blackboxcv.git

Move into the project directory:

cd blackboxcv

Install required dependencies:

pip install -r requirements.txt
Running the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser where you can:

Paste a job description

Upload a resume in PDF format

View the evaluation results

Future Improvements

Possible enhancements for the project include:

Automatic skill extraction from job descriptions

Skill gap analysis and recommendations

Visualization of resume match metrics

Larger dataset of institutional signals

Integration with real job postings

Author

Anuj Kumar Mishra
B.Tech Computer Science & Engineering
GL Bajaj Institute of Technology & Management