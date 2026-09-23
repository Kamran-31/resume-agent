import sys
import types

# Compatibility shim: Mock pkg_resources for CrewAI telemetry on minimal cloud environments
try:
    import pkg_resources
except ImportError:
    class _MockRequirement:
        @staticmethod
        def parse(s):
            return s

    mock_pkg = types.ModuleType("pkg_resources")
    mock_pkg.Requirement = _MockRequirement
    mock_pkg.get_distribution = lambda x: types.SimpleNamespace(version="0.80.0")
    mock_pkg.working_set = []
    sys.modules["pkg_resources"] = mock_pkg

import io
import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, LLM

# Disable CrewAI telemetry completely to avoid background network/package lookups
import os
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# ---------------------------------------------------------
# Page Configuration & Emerald / Violet Custom Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Review Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling using Emerald Green (#059669), Deep Violet (#4F46E5), and Warm Amber (#D97706)
st.markdown(
    """
    <style>
    /* Main container background accents */
    .stApp {
        background-color: #FAF9F6;
    }
    
    /* Top Banner / Hero Card */
    .hero-container {
        background: linear-gradient(135deg, #059669 0%, #4F46E5 100%);
        padding: 2rem;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Privacy Card */
    .privacy-notice {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 0.85rem 1.2rem;
        border-radius: 6px;
        color: #78350F;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* Custom button overrides */
    .stButton > button {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.8rem !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #047857 !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
    }

    /* Section Cards */
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_text_from_pdf(uploaded_file) -> str:
    """Safely extracts text content from an uploaded PDF file."""
    try:
        pdf_stream = io.BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_stream)
        extracted_pages = []

        if len(reader.pages) == 0:
            return ""

        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                extracted_pages.append(text.strip())

        return "\n\n".join(extracted_pages)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF document: {str(e)}")


def run_resume_review(resume_content: str, job_desc: str, api_key: str, model_id: str) -> str:
    """Initializes the CrewAI agent, task, and crew to analyze the resume."""
    # Initialize the Groq LLM using CrewAI's native LLM wrapper
    llm = LLM(
        model=f"groq/{model_id}",
        api_key=api_key,
        temperature=0.1,  # Low temperature for objective, deterministic parsing
    )

    # Single Agent Setup
    resume_reviewer = Agent(
        role="Senior Technical Recruiter and Career Strategist",
        goal="Produce an objective, evidence-based resume review against target job specifications without inventing details.",
        backstory=(
            "You are a strict, meticulous technical recruiter and career coach. "
            "You operate under an absolute zero-hallucination policy. "
            "You never invent, assume, or infer skills, certifications, work history, "
            "or educational qualifications not explicitly present in the candidate's resume. "
            "If an item is ambiguous or unstated, you classify it strictly as 'Unknown / Not Demonstrated'."
        ),
        llm=llm,
        verbose=False,
        memory=False,
        max_iter=1,
    )

    # Strict Output Task Prompt
    review_task = Task(
        description=(
            "Carefully analyze the supplied Resume and Target Job Description.\n\n"
            "=== RESUME ===\n"
            "{resume_text}\n\n"
            "=== TARGET JOB DESCRIPTION ===\n"
            "{job_description}\n\n"
            "RULES:\n"
            "1. Only state a requirement is met if it is explicitly documented in the resume.\n"
            "2. If an item cannot be determined directly from the text, label it as 'Unknown / Not Demonstrated'.\n"
            "3. Do not invent any experience, qualifications, or company history.\n\n"
            "Produce the final review structured EXACTLY with these 9 Markdown headings:\n"
            "### 1. Match Summary\n"
            "### 2. Skills Found\n"
            "### 3. Missing Requirements\n"
            "### 4. Unclear / Not Demonstrated\n"
            "### 5. Experience Gaps\n"
            "### 6. Education / Qualification Gaps\n"
            "### 7. Resume Improvements\n"
            "### 8. Keywords to Consider\n"
            "### 9. Priority Action Plan\n"
        ),
        expected_output="A structured 9-section markdown report comparing the resume against the job description.",
        agent=resume_reviewer,
    )

    crew = Crew(
        agents=[resume_reviewer],
        tasks=[review_task],
        verbose=False,
    )

    result = crew.kickoff(
        inputs={
            "resume_text": resume_content,
            "job_description": job_desc,
        }
    )

    return str(result)


# ---------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------

# Hero Banner
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎯 AI Resume Review Agent</div>
        <div class="hero-subtitle">
            Reliable, evidence-based resume benchmarking powered by CrewAI & Groq.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Privacy Notice
st.markdown(
    """
    <div class="privacy-notice">
        <strong>🔒 Privacy Notice:</strong> No resume or job data is stored permanently on our servers or databases. 
        Your inputs are held temporarily in session memory and transmitted securely to your configured Groq model for analysis.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Secret & Configuration Diagnostics
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    api_key_configured = "GROQ_API_KEY" in st.secrets and bool(st.secrets["GROQ_API_KEY"].strip())
    
    if api_key_configured:
        st.success("✅ Groq API Key Detected")
    else:
        st.error("❌ Groq API Key Missing in Secrets")

    groq_model_name = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")
    st.info(f"🤖 **Model:** `{groq_model_name}`")

    st.markdown("---")
    st.markdown(
        """
        **Audit Guarantees:**
        - Zero Hallucination Policy
        - Explicit Evidence Check
        - Gap Identification
        """
    )

# Input Layout
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.subheader("📄 Candidate Resume")
    input_method = st.radio(
        "Choose resume source:",
        ["Upload PDF", "Paste Plain Text"],
        horizontal=True,
    )

    resume_text = ""
    if input_method == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload resume file (.pdf)", type=["pdf"])
        if uploaded_pdf is not None:
            with st.spinner("Extracting text from PDF..."):
                try:
                    resume_text = extract_text_from_pdf(uploaded_pdf)
                    if not resume_text:
                        st.warning(
                            "⚠️ The uploaded PDF has no extractable text. "
                            "It might be a scanned image. Please paste the text manually."
                        )
                    else:
                        st.success(f"Extracted {len(resume_text.split())} words successfully.")
                except Exception as ex:
                    st.error(f"Error reading file: {str(ex)}")
    else:
        resume_text = st.text_area(
            "Paste resume text:",
            height=280,
            placeholder="Paste complete resume text here...",
        )

with col2:
    st.subheader("📋 Target Job Description")
    job_description = st.text_area(
        "Paste job description:",
        height=320,
        placeholder="Paste requirements, qualifications, and role responsibilities here...",
    )

st.markdown("---")

# Execution Action
start_review = st.button("🚀 Analyze Resume Fit", use_container_width=True)

if start_review:
    # 1. Validation Checks
    if not api_key_configured:
        st.error(
            "Cannot proceed: `GROQ_API_KEY` is missing. "
            "Please configure your key in `.streamlit/secrets.toml` or Streamlit Cloud Secrets."
        )
    elif not resume_text or not resume_text.strip():
        st.error("Please supply a valid resume (via PDF or text paste) before running the review.")
    elif not job_description or not job_description.strip():
        st.error("Please paste the target job description to benchmark against.")
    else:
        # 2. Execution with User-Friendly Diagnostics
        with st.spinner("Analyzing candidate profile against job specifications..."):
            try:
                groq_key = st.secrets["GROQ_API_KEY"].strip()
                review_report = run_resume_review(
                    resume_content=resume_text,
                    job_desc=job_description,
                    api_key=groq_key,
                    model_id=groq_model_name,
                )

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("## 📊 Comprehensive Resume Assessment")
                st.markdown(review_report)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                err_msg = str(e).lower()
                if "rate limit" in err_msg or "429" in err_msg:
                    st.error("⏱️ Groq API rate limit reached. Please wait 30-60 seconds and retry.")
                elif "authentication" in err_msg or "401" in err_msg or "invalid api key" in err_msg:
                    st.error("🔑 Invalid Groq API key. Check your secrets configuration.")
                elif "timeout" in err_msg:
                    st.error("⌛ Request timed out while awaiting the LLM. Try testing with a shorter prompt.")
                else:
                    st.error(
                        "An unexpected error occurred while executing the review. "
                        "Please verify your model name and network connectivity."
                    )
