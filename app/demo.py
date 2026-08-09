"""
demo.py
A polished Gradio demo: upload a photo, see the generated caption and a
Grad-CAM 'viewfinder' showing where the model looked.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from model import load_blip_model  # noqa: E402
from inference import generate_caption  # noqa: E402
from xai import generate_gradcam, overlay_heatmap  # noqa: E402

# ---------------------------------------------------------------------------
# Model (loaded once at startup)
# ---------------------------------------------------------------------------
processor, model, device = load_blip_model()
MODEL_NAME = "Salesforce/blip-image-captioning-base"


def caption_and_explain(image: Image.Image):
    if image is None:
        return (
            "<div class='caption-empty'>Upload a photo to see its caption here.</div>",
            None,
        )

    temp_path = "temp_upload.jpg"
    image.convert("RGB").save(temp_path)

    caption = generate_caption(temp_path, model, processor, device)

    cam, _, raw_image = generate_gradcam(temp_path, model, processor, device)
    overlay = overlay_heatmap(raw_image, cam)
    overlay_image = Image.fromarray(overlay)

    os.remove(temp_path)

    caption_html = f"""
    <div class="caption-card">
        <span class="caption-mark">&ldquo;</span>
        <span class="caption-text">{caption}</span>
        <span class="caption-mark caption-mark-end">&rdquo;</span>
    </div>
    """
    return caption_html, overlay_image


# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0E1420",
    background_fill_primary="#161D2E",
    background_fill_secondary="#0E1420",
    border_color_primary="#262F45",
    block_background_fill="#161D2E",
    block_border_color="#262F45",
    block_label_text_color="#8D95AC",
    block_title_text_color="#EDEFF4",
    body_text_color="#EDEFF4",
    body_text_color_subdued="#8D95AC",
    button_primary_background_fill="#E3A857",
    button_primary_background_fill_hover="#EFC07E",
    button_primary_text_color="#161014",
    input_background_fill="#0E1420",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
}

/* ---------- Header ---------- */
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #E3A857;
    margin-bottom: 10px;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 42px;
    line-height: 1.1;
    color: #EDEFF4;
    margin: 0 0 12px 0;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 15.5px;
    color: #8D95AC;
    max-width: 560px;
    line-height: 1.55;
    margin-bottom: 6px;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, #E3A857 0%, #262F45 40%);
    margin: 22px 0 28px 0;
    border: none;
}

/* ---------- Viewfinder frame (signature element) ---------- */
.viewfinder {
    position: relative;
    border-radius: 4px !important;
}
.viewfinder::before, .viewfinder::after,
.viewfinder .vf-tl, .viewfinder .vf-br {
    content: "";
}
.viewfinder {
    padding: 3px !important;
}
.viewfinder > * {
    border-radius: 2px !important;
}
.viewfinder::before {
    position: absolute;
    top: -1px; left: -1px;
    width: 22px; height: 22px;
    border-top: 2px solid #E3A857;
    border-left: 2px solid #E3A857;
    z-index: 5;
    pointer-events: none;
}
.viewfinder::after {
    position: absolute;
    bottom: -1px; right: -1px;
    width: 22px; height: 22px;
    border-bottom: 2px solid #E3A857;
    border-right: 2px solid #E3A857;
    z-index: 5;
    pointer-events: none;
}

/* ---------- Section labels ---------- */
.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8D95AC;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #E3A857;
    display: inline-block;
}

/* ---------- Caption card ---------- */
.caption-card {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 21px;
    line-height: 1.5;
    color: #EDEFF4;
    background: #0E1420;
    border: 1px solid #262F45;
    border-left: 3px solid #E3A857;
    border-radius: 4px;
    padding: 22px 26px;
    min-height: 90px;
}
.caption-mark {
    font-family: 'Fraunces', serif;
    color: #E3A857;
    font-size: 26px;
    font-style: normal;
}
.caption-empty {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: #556077;
    padding: 30px 4px;
    text-align: center;
}

/* ---------- Footer ---------- */
.footer-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    color: #556077;
    margin-top: 30px;
    padding-top: 18px;
    border-top: 1px solid #1B2338;
    line-height: 1.8;
}
.footer-note b { color: #8D95AC; font-weight: 500; }

/* ---------- Button ---------- */
button.primary {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
}
"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(theme=THEME, css=CSS, title="What the Model Sees") as demo:

    gr.HTML(
        """
        <div class="hero-eyebrow">IMAGE CAPTIONING &middot; EXPLAINABILITY DEMO</div>
        <div class="hero-title">What the model sees</div>
        <div class="hero-subtitle">
            Upload a photo. A vision-language model describes it in one sentence —
            and a Grad-CAM overlay shows exactly which region it looked at to decide.
        </div>
        <hr class="hero-divider" />
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label"><span class="dot"></span>01 &nbsp;INPUT PHOTO</div>')
            input_image = gr.Image(
                type="pil",
                label=None,
                show_label=False,
                elem_classes="viewfinder",
                height=340,
            )
            run_btn = gr.Button("Reveal the caption", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label"><span class="dot"></span>02 &nbsp;GENERATED CAPTION</div>')
            caption_output = gr.HTML(
                "<div class='caption-empty'>Your caption will appear here.</div>"
            )

            gr.HTML('<div class="panel-label" style="margin-top:22px"><span class="dot"></span>03 &nbsp;WHERE IT LOOKED &mdash; GRAD-CAM</div>')
            gradcam_output = gr.Image(
                label=None,
                show_label=False,
                elem_classes="viewfinder",
                height=260,
            )

    run_btn.click(
        fn=caption_and_explain,
        inputs=input_image,
        outputs=[caption_output, gradcam_output],
    )

    gr.HTML(
        f"""
        <div class="footer-note">
            <b>Model</b> &nbsp;{MODEL_NAME} (zero-shot, greedy decoding) &nbsp;&middot;&nbsp;
            <b>Explainability</b> &nbsp;Grad-CAM on the vision encoder &nbsp;&middot;&nbsp;
            <b>Device</b> &nbsp;{str(device).upper()}
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)