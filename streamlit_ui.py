
"""VisionLens — Deep Learning Image Intelligence Studio."""

import html

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models
from torchvision.models import ResNet101_Weights


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VisionLens | Image Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL DESIGN SYSTEM
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --background: #0B1020;
        --surface: #141C30;
        --surface-light: #1B2740;
        --border: #293650;
        --primary: #6C8CFF;
        --secondary: #A78BFA;
        --text: #F4F7FF;
        --muted: #A6B2CA;
        --success: #55D6BE;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, #18264A 0%, transparent 38%),
            radial-gradient(ellipse at 95% 10%, #251B48 0%, transparent 30%),
            var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.85);
    }

    .block-container {
        max-width: 1350px;
        padding: 2rem 2rem 3rem 2rem;
    }

    [data-testid="stSidebar"] {
        background: #0D1426;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.4px;
    }

    p, li, label {
        color: var(--muted);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 36px 34px;
        border: 1px solid #35456D;
        border-radius: 25px;
        background:
            radial-gradient(circle at 85% 15%, rgba(167,139,250,.25), transparent 35%),
            linear-gradient(120deg, #172747 0%, #20264B 55%, #29204D 100%);
        margin-bottom: 28px;
    }

    .eyebrow {
        color: #A9BFFF;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(34px, 5vw, 52px);
        font-weight: 700;
        line-height: 1.12;
        letter-spacing: -2px;
        color: #FFFFFF;
        margin-bottom: 12px;
    }

    .hero-title span {
        background: linear-gradient(90deg, #8CB4FF, #C4A7FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        max-width: 650px;
        color: #CFD8EF;
        font-size: 16px;
        line-height: 1.8;
    }

    .hero-chip {
        display: inline-block;
        padding: 7px 12px;
        margin-top: 16px;
        margin-right: 8px;
        border-radius: 30px;
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.14);
        color: #E4E9FF;
        font-size: 12px;
        font-weight: 600;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 6px;
    }

    .section-description {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    .surface-card {
        padding: 22px;
        background: linear-gradient(145deg, #182239, #131C30);
        border: 1px solid var(--border);
        border-radius: 18px;
        margin-bottom: 16px;
    }

    .feature-icon {
        font-size: 26px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #F4F7FF;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .feature-description {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.8;
    }

    .mini-label {
        color: #9AA9C5;
        font-size: 11px;
        letter-spacing: 1.2px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .mini-value {
        color: #F4F7FF;
        font-size: 17px;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .result-card {
        padding: 18px;
        margin-bottom: 13px;
        border: 1px solid #344466;
        border-radius: 16px;
        background: linear-gradient(120deg, #1B2A48, #1A203B);
    }

    .result-rank {
        color: #9DB5FF;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .result-name {
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 21px;
        font-weight: 700;
        margin-top: 5px;
        overflow-wrap: anywhere;
    }

    .result-score {
        color: #AFC2FF;
        font-size: 14px;
        font-weight: 600;
        margin-top: 5px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(20, 28, 48, .85);
        border: 1px dashed #6378AE;
        border-radius: 17px;
        padding: 14px;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent;
    }

    div[data-testid="stFileUploader"] button {
        background: #263A68;
        color: white;
        border: 1px solid #536DA8;
        border-radius: 9px;
    }

    div.stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: 1px solid #849DFF;
        background: linear-gradient(100deg, #5278F5, #8869E8);
        color: white;
        font-weight: 700;
        transition: all .2s ease;
    }

    div.stButton > button:hover {
        border-color: #C0CBFF;
        color: white;
        filter: brightness(1.12);
    }

    [data-testid="stMetric"] {
        padding: 18px;
        background: #172239;
        border: 1px solid var(--border);
        border-radius: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: #F4F7FF;
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #638BFF, #A78BFA);
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    .footer {
        text-align: center;
        color: #8290AB;
        font-size: 12px;
        line-height: 1.8;
        padding-top: 12px;
    }

    hr {
        border-color: var(--border);
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }

        .hero {
            padding: 25px 20px;
        }

        .hero-title {
            letter-spacing: -1px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🔍 VisionLens")
    st.caption("IMAGE INTELLIGENCE STUDIO")
    st.divider()

    st.markdown("### 🧠 Model details")
    st.markdown("**Architecture:** ResNet-101")
    st.markdown("**Framework:** PyTorch")
    st.markdown("**Categories:** ImageNet")
    st.markdown("**Predictions:** Top 5")

    st.divider()

    st.markdown("### ⚙️ Workflow")
    st.markdown(
        """
        **01 — Upload**  
        Select a JPG or PNG image.

        **02 — Process**  
        Prepare the image for the model.

        **03 — Predict**  
        Run the pretrained neural network.

        **04 — Explore**  
        Review the top five predictions.
        """
    )

    st.divider()

    st.markdown("### 📌 About")
    st.caption(
        "An educational image-classification application demonstrating "
        "transfer learning through a pretrained ResNet-101 model."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">DEEP LEARNING · COMPUTER VISION</div>
        <div class="hero-title">See beyond the <span>image.</span></div>
        <div class="hero-description">
            Welcome to VisionLens, an image intelligence studio that uses
            a pretrained deep learning model to identify likely objects
            and visual categories in your pictures.
        </div>
        <span class="hero-chip">✦ ResNet-101</span>
        <span class="hero-chip">⚡ PyTorch</span>
        <span class="hero-chip">◈ ImageNet</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING AND PREDICTION
# =========================================================

@st.cache_resource
def load_model():
    """Load pretrained ResNet-101, preprocessing and ImageNet categories."""
    weights = ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights)
    model.eval()

    return model, weights.transforms(), weights.meta["categories"]


def predict(image):
    """Return the five highest-probability ImageNet categories."""
    model, preprocess, categories = load_model()

    image = image.convert("RGB")
    batch = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        output = model(batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        scores, indices = torch.topk(probabilities, k=5)

    return [
        (categories[index.item()], score.item() * 100)
        for score, index in zip(scores, indices)
    ]


# =========================================================
# MAIN WORKSPACE
# =========================================================

st.markdown(
    '<div class="section-heading">Image analysis workspace</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Upload an image to explore the categories identified by the model.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="surface-card">
            <div class="feature-icon">📸</div>
            <div class="feature-title">Your next image starts here</div>
            <div class="feature-description">
                Upload a picture using the area above. VisionLens will show
                an image preview and the model's five highest-scoring
                ImageNet categories.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What you can explore")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:
        st.markdown(
            """
            <div class="surface-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Deep Learning</div>
                <div class="feature-description">
                    Uses a pretrained ResNet-101 convolutional neural network
                    for image classification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature2:
        st.markdown(
            """
            <div class="surface-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Top 5 predictions</div>
                <div class="feature-description">
                    Explore five possible categories ranked by the model's
                    output probabilities.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature3:
        st.markdown(
            """
            <div class="surface-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Confidence scores</div>
                <div class="feature-description">
                    Compare model probabilities with clear percentages
                    and visual progress indicators.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        image_column, result_column = st.columns(
            [1, 1.15],
            gap="large",
        )

        with image_column:
            st.markdown("### 🖼️ Image preview")
            st.image(image, use_container_width=True)

            info1, info2 = st.columns(2)

            with info1:
                st.markdown(
                    f"""
                    <div class="surface-card">
                        <div class="mini-label">Image width</div>
                        <div class="mini-value">{image.width} px</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with info2:
                st.markdown(
                    f"""
                    <div class="surface-card">
                        <div class="mini-label">Image height</div>
                        <div class="mini-value">{image.height} px</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(f"File: {html.escape(uploaded_file.name)}")

        with result_column:
            st.markdown("### ✨ Classification results")

            if st.button(
                "🚀 Analyse image",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Preparing the model and analysing your image..."
                    ):
                        predictions = predict(image)

                    st.session_state["visionlens_predictions"] = predictions
                    st.session_state["visionlens_file_id"] = uploaded_file.file_id

                except Exception as error:
                    st.error(
                        "Classification failed. Check your internet connection "
                        "and the terminal for details."
                    )
                    st.exception(error)

            saved_file_id = st.session_state.get("visionlens_file_id")
            saved_predictions = st.session_state.get("visionlens_predictions")

            if saved_file_id == uploaded_file.file_id and saved_predictions:
                best_label, best_score = saved_predictions[0]

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-rank">🏆 TOP PREDICTION</div>
                        <div class="result-name">
                            {html.escape(best_label)}
                        </div>
                        <div class="result-score">
                            Model probability: {best_score:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                metric1, metric2 = st.columns(2)

                with metric1:
                    st.metric("Best match", f"{best_score:.2f}%")

                with metric2:
                    st.metric("Categories shown", len(saved_predictions))

                st.markdown("#### Other likely categories")

                for rank, (label, score) in enumerate(
                    saved_predictions,
                    start=1,
                ):
                    st.markdown(
                        f"""
                        <div class="mini-label">
                            PREDICTION {rank}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"**{html.escape(label)}** &nbsp; "
                        f"<span style='color:#AFC2FF;font-size:13px'>"
                        f"{score:.2f}%</span>",
                        unsafe_allow_html=True,
                    )

                    st.progress(max(0.0, min(score / 100, 1.0)))

                st.info(
                    "Interpretation: these values are the model's predicted "
                    "probabilities over ImageNet classes. They are not a "
                    "guarantee that a prediction is correct."
                )

            else:
                st.markdown(
                    """
                    <div class="surface-card">
                        <div class="feature-icon">🔬</div>
                        <div class="feature-title">Ready to analyse</div>
                        <div class="feature-description">
                            Click the Analyse image button to run the model
                            and view its five highest-scoring categories.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except (UnidentifiedImageError, OSError):
        st.error(
            "This image could not be opened. Please upload a valid JPG or PNG."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>🔍 VisionLens</strong><br>
        Deep Learning Image Classification Studio<br>
        Built with Streamlit · PyTorch · ResNet-101<br>
        Educational demonstration — predictions may be incorrect.
    </div>
    """,
    unsafe_allow_html=True,
)

