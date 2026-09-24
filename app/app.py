"""VeggieVision — single-page vegetable image classifier, farmers-market theme.

A clean, self-contained Streamlit app:
  - full-page market background photo (assets/images (1).jpeg) with a warm
    overlay so text stays readable
  - upload a photo (styled dropzone) -> the image replaces the dropzone
  - a "Predict" button sits right under the photo
  - the real prediction (class + confidence bar) appears beside the photo
  - "Upload another photo" clears the current one and starts over

Predictions ALWAYS come from the trained CNN (models/vegetable_cnn.h5); there
is no random/stub fallback. If the model cannot be loaded the app explains why
instead of inventing a result.

Assets:
  The background photo lives at assets/images (1).jpeg (project assets root,
  as defined in src/config.py). A few common name variants are also checked
  automatically. If it isn't found, the app falls back to a plain warm
  gradient instead of crashing.

Run:
    streamlit run app/app.py
"""

import base64
import io
import os
import sys
import tempfile
from pathlib import Path

# Make the project root importable no matter where `streamlit run` is invoked,
# so `src.*` (the model engine) resolves consistently.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from PIL import Image, ImageOps

from src.config import ASSETS_DIR

# --------------------------------------------------------------------------- #
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Vegetable Classifier",
    page_icon="🥕",
    layout="centered",          # prevents edge-to-edge stretching on wide screens
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Constants & brand palette — warm farmers-market theme
# --------------------------------------------------------------------------- #

IMG_EXTS = ["jpg", "jpeg", "png", "bmp"]
BOX_SIZE = 400                  # px, the upload box / preview box

BG_CANDIDATES = [
    "images (1).jpeg", "images (1).jpg", "images(1).jpeg", "images(1).jpg",
    "market_bg.jpeg", "market_bg.jpg",
]

WOOD       = "#6B4226"           # crate / stall wood
WOOD_DK    = "#4A2E1A"
GREEN      = "#3F7D32"           # fresh produce green
GREEN_DK   = "#2C5A22"
ORANGE     = "#E08E1D"           # carrot / turmeric accent
CREAM      = "#FFFCF5"           # card background (glassy)
CHARCOAL   = "#2B2118"           # main text (warm dark brown-black)
MUTED      = "#6E5D4C"
BORDER     = "rgba(107, 66, 38, 0.18)"
FONT       = "'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"


def _load_background_b64() -> str | None:
    """Return the market background image as base64, or None if not found."""
    for name in BG_CANDIDATES:
        path = ASSETS_DIR / name
        if path.is_file():
            try:
                return base64.b64encode(path.read_bytes()).decode()
            except Exception:
                continue
    return None


BG_B64 = _load_background_b64()

if BG_B64:
    _page_background = (
        f'linear-gradient(180deg, rgba(35,22,12,0.72) 0%, '
        f'rgba(20,28,16,0.72) 55%, rgba(15,20,12,0.82) 100%), '
        f'url("data:image/jpeg;base64,{BG_B64}")'
    )
    _bg_extra = "background-size: cover; background-position: center; " \
                "background-attachment: fixed;"
else:
    # Graceful fallback if the photo hasn't been added to assets/ yet.
    _page_background = f"linear-gradient(180deg, {WOOD} 0%, {GREEN_DK} 100%)"
    _bg_extra = ""

# --------------------------------------------------------------------------- #
# Global styles
# --------------------------------------------------------------------------- #

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:wght@600;700&display=swap');

    html, body, .stApp {{
        font-family: {FONT};
        color: {CHARCOAL};
        background-image: {_page_background};
        {_bg_extra}
    }}

    /* Remove the visible scrollbar (page still scrolls with wheel/trackpad) */
    * {{ scrollbar-width: none; -ms-overflow-style: none; }}
    *::-webkit-scrollbar {{ display: none; width: 0; height: 0; }}

    #MainMenu, header, footer {{ visibility: hidden; }}
    .block-container {{ max-width: 720px; padding-top: 2rem; padding-bottom: 2rem; }}

    /* ---- glass card helper ---- */
    .glass {{
        background: rgba(255, 252, 245, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(20, 15, 8, 0.25);
        backdrop-filter: blur(6px);
    }}

    /* ---- market sign header ---- */
    .market-sign {{
        text-align: center;
        padding: 22px 24px 20px;
        margin-bottom: 22px;
    }}
    .badge {{
        display: inline-block;
        background: {GREEN};
        color: #fff;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 999px;
        margin-bottom: 12px;
    }}
    .title {{
        font-family: 'Fraunces', {FONT};
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: {WOOD_DK};
        margin: 0 0 6px 0;
    }}
    .subtitle {{
        color: {MUTED};
        font-size: 1.02rem;
        margin: 0;
    }}

    /* ---- upload section wrapper ---- */
    .upload-wrap {{
        padding: 26px;
        margin-bottom: 18px;
        text-align: center;
    }}
    .upload-wrap h4 {{
        margin: 0 0 14px 0;
        font-size: 0.95rem;
        font-weight: 700;
        color: {WOOD_DK};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    /* =========================================================== */
    /* Restyle Streamlit's REAL file-uploader dropzone in place.    */
    /* We do NOT hide/duplicate the widget — this is what makes the */
    /* box reliably clickable (the earlier overlay trick could      */
    /* misalign and block clicks; this approach can't).             */
    /* =========================================================== */
    [data-testid="stFileUploader"] {{
        width: {BOX_SIZE}px;
        max-width: 100%;
        margin: 0 auto;
    }}
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {{
        position: relative;
        width: 100%;
        min-height: {BOX_SIZE}px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        border: 3px dashed {GREEN};
        border-radius: 20px;
        background: repeating-linear-gradient(
            135deg, rgba(63,125,50,0.05) 0px, rgba(63,125,50,0.05) 10px,
            rgba(63,125,50,0.00) 10px, rgba(63,125,50,0.00) 20px
        ), #FFFEFA;
        transition: border-color .2s ease, background .2s ease, transform .15s ease;
        overflow: hidden;
        cursor: pointer;
    }}
    [data-testid="stFileUploader"] section:hover,
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {ORANGE};
        transform: translateY(-1px);
    }}
    /* custom icon layered on top, purely decorative: pointer-events:none
       guarantees it can never block a click from reaching the real widget */
    [data-testid="stFileUploader"] section::before,
    [data-testid="stFileUploaderDropzone"]::before {{
        content: "🧺";
        font-size: 54px;
        pointer-events: none;
        margin-bottom: 4px;
    }}
    /* keep the real instructions text, just recolor / rescale it */
    [data-testid="stFileUploader"] section svg {{
        display: none;                 /* default cloud icon -> replaced by 🧺 above */
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: {CHARCOAL} !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] span {{
        font-size: 15px !important;
        font-weight: 600 !important;
        color: {CHARCOAL} !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] small {{
        color: {MUTED} !important;
    }}
    [data-testid="stFileUploader"] button {{
        border-radius: 999px !important;
        border: 1px solid {GREEN} !important;
        color: {GREEN_DK} !important;
        background: {CREAM} !important;
        font-weight: 700 !important;
        margin-top: 6px !important;
    }}
    [data-testid="stFileUploader"] button:hover {{
        background: {GREEN} !important;
        color: #fff !important;
    }}
    /* uploaded-file chip row (name / size / remove) — keep, just tone down */
    [data-testid="stFileUploaderFile"] {{
        background: {CREAM} !important;
        border-radius: 10px !important;
        border: 1px solid {BORDER} !important;
    }}

    /* ---- preview card (image replaces the upload square once chosen) ---- */
    .preview-card {{
        padding: 18px;
        margin-bottom: 14px;
        text-align: center;
    }}
    .preview-img {{
        width: 100%;
        max-width: {BOX_SIZE}px;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: 16px;
        border: 4px solid {CREAM};
        box-shadow: 0 8px 22px rgba(20, 15, 8, 0.30);
    }}
    .img-cap {{
        margin-top: 10px;
        font-size: 0.8rem;
        font-weight: 600;
        color: {MUTED};
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .dropzone-hint {{
        text-align: center;
        font-size: 0.82rem;
        color: {CREAM};
        margin-top: 14px;
        text-shadow: 0 1px 3px rgba(0,0,0,0.5);
    }}

    /* ---- "nothing predicted yet" placeholder card ---- */
    .hint-card {{
        padding: 40px 22px;
        height: 100%;
        min-height: 320px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }}
    .hint-card .hint-ico {{ font-size: 44px; margin-bottom: 10px; }}
    .hint-card .hint-txt {{ color: {MUTED}; font-size: 0.95rem; max-width: 240px; }}

    /* ---- buttons ---- */
    .stButton > button {{
        border-radius: 999px;
        font-weight: 700;
        font-family: {FONT};
        transition: all .2s ease;
    }}
    .stButton > button:focus {{ box-shadow: none; }}

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {{
        background: {ORANGE};
        border: 1px solid {ORANGE};
        color: #fff;
        padding: 0.7rem 1.8rem;
        font-size: 1.0rem;
        box-shadow: 0 6px 16px rgba(224, 142, 29, 0.35);
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {{
        background: #C67912;
        border-color: #C67912;
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(224, 142, 29, 0.45);
    }}
    .stButton > button[kind="primary"]:disabled {{
        background: #E9C48F;
        border-color: #E9C48F;
        color: #fff;
        box-shadow: none;
    }}

    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {{
        background: transparent;
        border: none;
        color: {CREAM};
        font-size: 0.82rem;
        text-decoration: underline;
    }}
    .stButton > button[kind="secondary"]:hover {{
        color: {ORANGE};
        background: transparent;
    }}

    .stSpinner > div {{ color: {ORANGE}; }}

    /* ---- result card ---- */
    .result-card {{
        padding: 24px 26px 26px;
        margin-top: 20px;
        text-align: center;
    }}
    .result-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {MUTED};
        margin-bottom: 4px;
    }}
    .result-name {{
        font-family: 'Fraunces', {FONT};
        font-size: 2rem;
        font-weight: 700;
        color: {WOOD_DK};
        text-transform: capitalize;
    }}
    .conf-row {{
        display: flex;
        justify-content: center;
        gap: 8px;
        align-items: baseline;
        margin: 16px 0 8px 0;
        font-size: 0.92rem;
        color: {MUTED};
    }}
    .conf-value {{ font-weight: 800; color: {CHARCOAL}; }}
    .conf-bar {{
        height: 10px;
        width: 100%;
        background: #EFE7DA;
        border-radius: 999px;
        overflow: hidden;
    }}
    .conf-fill {{
        height: 100%;
        background: linear-gradient(90deg, {GREEN}, {ORANGE});
        border-radius: 999px;
        transition: width .7s ease;
    }}
    .demo-note {{
        text-align: center; font-size: 0.78rem; color: {MUTED};
        margin-top: 12px;
    }}

    /* ---- footer ---- */
    .footer {{
        text-align: center;
        color: {CREAM};
        font-size: 12px;
        margin-top: 40px;
        padding: 14px 0;
        text-shadow: 0 1px 3px rgba(0,0,0,0.5);
    }}
</style>
"""

# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading the trained model...")
def _load_real_model():
    """Try to load the trained model bundle.

    Returns either {"model", "class_names"} or {"error": <message>} so the UI
    can explain why real predictions are unavailable.
    """
    try:
        from src import predict as engine
        model = engine.get_model()
        class_names = engine.get_class_names()
        return {"model": model, "class_names": class_names}
    except Exception as exc:                       # model missing / TF unavailable
        return {"error": str(exc)}


def load_model():
    """Return (bundle, error): the model bundle or None, plus an error message."""
    result = _load_real_model()
    error = result.get("error")
    if error is not None:
        return None, error
    return result, None


def _predict_with_real_model(bundle, image_bytes: bytes, filename: str) -> dict:
    """Classify with the real CNN (matches training preprocessing: raw pixels)."""
    from src import predict as engine  # local import keeps the app decoupled

    ext = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = engine.predict(
            tmp_path, model=bundle["model"], class_names=bundle["class_names"]
        )
    finally:
        os.unlink(tmp_path)
    return result


def predict(image_bytes: bytes, filename: str, bundle=None) -> dict:
    """Classify an image with the real CNN.

    Returns {"class_name", "confidence", ...} straight from the trained model.
    Never falls back to a random/stub result: without a loaded model it raises
    so the UI shows the real error instead of fabricating output.
    """
    if bundle is None:
        raise RuntimeError(
            "The trained model is not available. Place models/vegetable_cnn.h5 "
            "and models/class_names.json in the project and restart the app."
        )
    return _predict_with_real_model(bundle, image_bytes, filename)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def make_preview_b64(data: bytes) -> str | None:
    """Crop + resize to a BOX_SIZE square (no distortion), return base64 JPEG.

    Returns None if the bytes are not a decodable image, so the UI never
    crashes on a corrupted upload.
    """
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        fitted = ImageOps.fit(
            image, (BOX_SIZE, BOX_SIZE), method=Image.LANCZOS, centering=(0.5, 0.5)
        )
        buf = io.BytesIO()
        fitted.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def uploader_key():
    """Recompute key= each time 'Clear' is pressed, so the widget resets."""
    return f"veg_uploader_{st.session_state.get('clear_counter', 0)}"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def render_ui(bundle, model_error=None) -> None:
    # --- header ------------------------------------------------------- #
    st.markdown(
        f"""
        <div class="glass market-sign">
            <span class="badge">Image Classification</span>
            <div class="title">🥕 Vegetable Classifier</div>
            <div class="subtitle">Fresh from the stall to the model —
            upload a photo and see what's in the basket.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bundle is None:
        st.warning(
            "The trained model could not be loaded, so predictions are "
            "unavailable. Place `models/vegetable_cnn.h5` and "
            "`models/class_names.json` in the project and restart the app.\n\n"
            f"Reason: {model_error or 'unknown'}"
        )

    # The chosen image (bytes + original filename) survives reruns in
    # session_state, so the file_uploader can stop rendering once selected.
    active = st.session_state.get("active_img")

    # ---- step 1: no image yet -> show the styled dropzone ---------- #
    if active is None:
        uploaded = st.file_uploader(
            "Upload a vegetable image",
            type=IMG_EXTS,
            key=uploader_key(),
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.session_state["active_img"] = {
                "bytes": uploaded.getvalue(),
                "name": uploaded.name,
            }
            st.session_state.pop("prediction", None)
            st.rerun()
        st.markdown(
            '<div class="dropzone-hint">☝️ Tap the crate above to choose or '
            'drag &amp; drop a photo of your produce.</div>',
            unsafe_allow_html=True,
        )
        return

    # ---- step 2: image selected -> show photo + Predict beside result - #
    img_col, res_col = st.columns([3, 2], gap="large")

    with img_col:
        preview_b64 = make_preview_b64(active["bytes"])
        if preview_b64 is None:
            st.markdown(
                '<div class="glass hint-card"><div class="hint-ico">⚠️</div>'
                f'<div class="hint-txt">Could not preview <b>{active["name"]}</b> — '
                'the file doesn\'t look like a readable image.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="glass preview-card">
                    <img class="preview-img" src="data:image/jpeg;base64,{preview_b64}" />
                    <div class="img-cap">🖼️ {active["name"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            predict_clicked = st.button(
                "🔍 Predict",
                type="primary",
                width="stretch",
                disabled=(bundle is None),
            )

        _, clear_col, _ = st.columns([1, 2, 1])
        with clear_col:
            if st.button("⇆ Upload another photo", type="secondary"):
                st.session_state["clear_counter"] = (
                    st.session_state.get("clear_counter", 0) + 1
                )
                st.session_state.pop("active_img", None)
                st.session_state.pop("prediction", None)
                st.rerun()

    # ---- step 3: prediction ---------------------------------------- #
    if predict_clicked:
        with st.spinner("Classifying..."):
            try:
                result = predict(active["bytes"], active["name"], bundle=bundle)
            except Exception as exc:            # surface the real problem
                st.error(f"Prediction failed: {exc}")
                result = None
        if result is not None:
            st.session_state["prediction"] = result

    prediction = st.session_state.get("prediction")
    with res_col:
        if prediction is not None:
            _render_result(prediction)
        else:
            st.markdown(
                '<div class="glass hint-card">'
                '<div class="hint-ico">🧐</div>'
                '<div class="hint-txt">Your prediction will appear here once you '
                'press <b>Predict</b>.</div>'
                '</div>',
                unsafe_allow_html=True,
            )


def _render_result(result: dict) -> None:
    name = result["class_name"].replace("_", " ").title()
    conf = max(0.0, min(1.0, float(result["confidence"])))

    st.markdown(
        f"""
        <div class="glass result-card">
            <div class="result-label">Prediction</div>
            <div class="result-name">🥬 {name}</div>
            <div class="conf-row">
                <span>Confidence</span>
                <span class="conf-value">{conf * 100:.1f}%</span>
            </div>
            <div class="conf-bar">
                <div class="conf-fill" style="width:{conf * 100:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if BG_B64 is None:
        st.caption(
            "Tip: add your market photo as "
            f"`{ASSETS_DIR}/images (1).jpeg` to enable the background."
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    bundle, model_error = load_model()
    render_ui(bundle, model_error)

    st.markdown(
        '<div class="footer">🥕 VeggieVision v2.0 · Streamlit · '
        'TensorFlow CNN · Vegetable Image Dataset</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()