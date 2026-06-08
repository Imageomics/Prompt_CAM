"""
Prompt-CAM Interactive Visualization App

Run this app to explore Prompt-CAM visualizations through a web interface
without writing any Python code:

    python app.py

Then open the URL shown in your terminal (typically http://localhost:7860).
"""

import io
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – must be set before pyplot import
import matplotlib.pyplot as plt

import gradio as gr
import torch
from dotwiz import DotWiz
from PIL import Image

from experiment.build_model import get_model
from utils.misc import load_yaml, set_seed
from utils.visual_utils import load_image, prune_and_plot_ranked_heads

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

# Map human-readable labels to YAML config paths
CONFIGS = {
    "DINO / CUB-200 (200 bird species)": "experiment/config/prompt_cam/dino/cub/args.yaml",
    "DINO / Stanford Cars (196 classes)": "experiment/config/prompt_cam/dino/car/args.yaml",
    "DINO / Stanford Dogs (120 breeds)": "experiment/config/prompt_cam/dino/dog/args.yaml",
    "DINO / Oxford Pets (37 breeds)": "experiment/config/prompt_cam/dino/pet/args.yaml",
    "DINO / Birds-525 (525 species)": "experiment/config/prompt_cam/dino/birds_525/args.yaml",
    "DINOv2 / CUB-200 (200 bird species)": "experiment/config/prompt_cam/dinov2/cub/args.yaml",
    "DINOv2 / Stanford Dogs (120 breeds)": "experiment/config/prompt_cam/dinov2/dog/args.yaml",
    "DINOv2 / Oxford Pets (37 breeds)": "experiment/config/prompt_cam/dinov2/pet/args.yaml",
}

# Dataset class counts (for informational display)
DATASET_CLASS_COUNTS = {
    "cub": 200,
    "car": 196,
    "dog": 120,
    "pet": 37,
    "birds_525": 525,
}

# Sample images bundled with the repository (CUB dataset, DINO backbone)
SAMPLE_IMAGES = {
    "Scott's Oriole (class 97)": ("samples/Scott_Oriole.jpg", 97),
    "Baltimore Oriole (class 94)": ("samples/Baltimore_Oriole.jpg", 94),
    "Orchard Oriole (class 96)": ("samples/Orchard_Oriole.jpg", 96),
    "Rusty Blackbird (class 10)": ("samples/rusty_Blackbird.jpg", 10),
    "Brewer's Blackbird (class 11)": ("samples/Brewer_Blackbird.jpg", 11),
    "Yellow-headed Blackbird (class 195)": ("samples/yellow_headed_blackbird.jpg", 195),
}

# ---------------------------------------------------------------------------
# Core visualization function
# ---------------------------------------------------------------------------

def visualize_traits(
    config_label: str,
    checkpoint_path: str,
    image_input,
    target_class: int,
    top_traits: int,
):
    """Load a checkpoint, run Prompt-CAM on *image_input*, and return the plot.

    Parameters
    ----------
    config_label:
        Human-readable label from the CONFIGS dictionary.
    checkpoint_path:
        Path (str) to a saved ``.pt`` checkpoint produced by ``main.py``.
    image_input:
        Either a PIL Image (from the upload widget) or a file path string.
    target_class:
        Class index whose prompts should be visualised.
    top_traits:
        Number of top attention heads (traits) to highlight.

    Returns
    -------
    result_image : PIL.Image | None
        The visualisation plot, or None on failure.
    status_message : str
        A short success / error message to display in the UI.
    """
    if checkpoint_path is None:
        return None, "❌ Please upload a model checkpoint (.pt file)."
    if image_input is None:
        return None, "❌ Please upload an image or select a sample image."
    if config_label not in CONFIGS:
        return None, "❌ Invalid model/dataset selection."

    config_path = CONFIGS[config_label]
    if not os.path.exists(config_path):
        return None, f"❌ Config file not found: {config_path}"

    try:
        # Build args from the YAML config
        yaml_config = load_yaml(config_path)
        args = DotWiz(yaml_config)

        args.checkpoint = checkpoint_path
        args.vis_cls = int(target_class)
        args.top_traits = int(top_traits)
        args.test_batch_size = 1
        args.random_seed = 42
        set_seed(args.random_seed)

        # Load the model (visualize=True skips loading raw pre-trained weights)
        model, _, _ = get_model(args, visualize=True)
        state = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        # Prepare the input image
        if isinstance(image_input, str):
            # File path (e.g. from sample image selection)
            sample = load_image(image_input)
        else:
            # PIL Image from the upload widget – save temporarily
            tmp_path = "/tmp/_prompt_cam_input.jpg"
            image_input.save(tmp_path)
            sample = load_image(tmp_path)
        sample = sample.to(args.device, non_blocking=True)

        # Run Prompt-CAM and capture the matplotlib figure
        plt.close("all")
        with torch.no_grad():
            prune_and_plot_ranked_heads(model, sample, int(target_class), args)
        fig = plt.gcf()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        result_image = Image.open(buf).copy()
        plt.close("all")

        return result_image, "✅ Visualisation complete!"

    except FileNotFoundError as exc:
        return None, f"❌ File not found: {exc}"
    except KeyError as exc:
        return None, (
            f"❌ Checkpoint is missing key {exc}. "
            "Make sure you are using a checkpoint saved by this codebase."
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"❌ Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Gradio interface helpers
# ---------------------------------------------------------------------------

def load_sample(sample_label: str):
    """Return the image path and suggested class index for a sample image."""
    if sample_label in SAMPLE_IMAGES:
        path, cls = SAMPLE_IMAGES[sample_label]
        if os.path.exists(path):
            return Image.open(path), cls
    return None, 0


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

def build_interface():
    with gr.Blocks(title="Prompt-CAM Visualizer") as demo:
        gr.Markdown(
            """
# 🔍 Prompt-CAM Interactive Visualizer

**Prompt-CAM** makes Vision Transformers interpretable for fine-grained analysis.
This app lets you explore *which traits* the model focuses on for any class —
no Python knowledge required!

### Quick start
1. Select a **Model / Dataset** combination.
2. Upload the matching **checkpoint** (`.pt` file downloaded from the links in the README).
3. Upload an **image** or pick one of the built-in sample images.
4. Set the **target class index** and **number of traits** to show.
5. Click **▶ Run Visualization**.
"""
        )

        with gr.Row():
            # ---- Left column: inputs ----------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model & Checkpoint")
                config_dropdown = gr.Dropdown(
                    choices=list(CONFIGS.keys()),
                    value="DINO / CUB-200 (200 bird species)",
                    label="Model / Dataset",
                    info="Choose the backbone and dataset that matches your checkpoint.",
                )
                checkpoint_upload = gr.File(
                    label="Checkpoint file (.pt)",
                    file_types=[".pt"],
                    type="filepath",
                )

                gr.Markdown("### 🖼️ Input Image")
                with gr.Tab("Upload your own image"):
                    image_upload = gr.Image(
                        label="Upload image",
                        type="pil",
                    )
                with gr.Tab("Use a sample image"):
                    sample_dropdown = gr.Dropdown(
                        choices=list(SAMPLE_IMAGES.keys()),
                        value=list(SAMPLE_IMAGES.keys())[0],
                        label="Sample image (CUB dataset)",
                    )
                    sample_preview = gr.Image(
                        label="Preview",
                        type="pil",
                        interactive=False,
                    )
                    load_sample_btn = gr.Button("Load sample →", size="sm")

                gr.Markdown("### 🎛️ Visualisation Parameters")
                target_class = gr.Number(
                    label="Target class index",
                    value=97,
                    precision=0,
                    info=(
                        "Zero-based class index to visualise. "
                        "For CUB-200: 0–199, Stanford Cars: 0–195, etc."
                    ),
                )
                top_traits = gr.Slider(
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=3,
                    label="Top traits to show",
                    info="Number of most important attention heads to highlight.",
                )

                run_btn = gr.Button("▶ Run Visualization", variant="primary")

            # ---- Right column: outputs --------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                output_image = gr.Image(
                    label="Trait visualisation",
                    type="pil",
                    interactive=False,
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )

        # ---- Wire up callbacks ------------------------------------------
        load_sample_btn.click(
            fn=load_sample,
            inputs=[sample_dropdown],
            outputs=[sample_preview, target_class],
        )

        # Visualisation can use either the uploaded image or the sample preview
        def run_with_either_image(
            config_label, checkpoint_path, uploaded_img, sample_img, target_cls, top_k
        ):
            image = uploaded_img if uploaded_img is not None else sample_img
            return visualize_traits(config_label, checkpoint_path, image, target_cls, top_k)

        run_btn.click(
            fn=run_with_either_image,
            inputs=[
                config_dropdown,
                checkpoint_upload,
                image_upload,
                sample_preview,
                target_class,
                top_traits,
            ],
            outputs=[output_image, status_text],
        )

        gr.Markdown(
            """
---
### 📖 Tips

- **Class index** – class numbering starts at **0**. For CUB-200, class 0 is
  *001.Black_footed_Albatross*, class 97 is *Scott's Oriole*, etc.
- **Checkpoint** – download checkpoints from the Google Drive links in the
  [README](README.md) and place them in `checkpoints/{model}/{dataset}/`.
- **GPU vs CPU** – a GPU speeds up inference considerably, but the app works on
  CPU as well (expect slower runtimes).
- **Sample images** – the bundled samples are from the CUB-200 validation set
  and work best with a CUB checkpoint.

See the [README](README.md) for full documentation and training instructions.
"""
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    interface = build_interface()
    interface.launch(share=False)
