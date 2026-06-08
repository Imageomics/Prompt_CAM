# Contributing to Prompt-CAM

Thank you for your interest in Prompt-CAM! This guide explains how to use the
repository and how to contribute to it. No advanced Python knowledge is required
to run the visualization tools — we've designed the workflow so that everyone can
explore the model's interpretability features.

---

## Table of Contents

1. [Using the Tools Without Python Knowledge](#1-using-the-tools-without-python-knowledge)
2. [Environment Setup](#2-environment-setup)
3. [Downloading Checkpoints](#3-downloading-checkpoints)
4. [Running the Web App (Recommended for New Users)](#4-running-the-web-app-recommended-for-new-users)
5. [Running the Demo Notebook](#5-running-the-demo-notebook)
6. [Running Visualization from the Command Line](#6-running-visualization-from-the-command-line)
7. [Training a New Model](#7-training-a-new-model)
8. [Extending the Codebase](#8-extending-the-codebase)
9. [Reporting Issues](#9-reporting-issues)
10. [Code Style](#10-code-style)

---

## 1. Using the Tools Without Python Knowledge

You do **not** need to write or edit any Python code to use the visualization
tools. Choose one of the following entry points:

| Entry point | Python knowledge needed | What it does |
|---|---|---|
| **Google Colab** [![Colab](https://img.shields.io/badge/Google_Colab-blue)](https://colab.research.google.com/drive/1co1P5LXSVb-g0hqv8Selfjq4WGxSpIFe?usp=sharing) | None (runs in the browser) | Interactive demo in the cloud — no local install needed |
| **`python app.py`** (web app) | None after setup | Point-and-click web interface for visualization |
| **`python visualize.py`** (CLI) | Basic command line | Visualize an entire class from a dataset |
| **`demo.ipynb`** (notebook) | Basic Jupyter skills | Interactive Python notebook |

For first-time users we recommend starting with the **Google Colab** link or
the **web app** described in [Section 4](#4-running-the-web-app-recommended-for-new-users).

---

## 2. Environment Setup

Perform these steps once on your local machine:

```bash
# 1. Create and activate a conda environment
conda create -n prompt_cam python=3.10
conda activate prompt_cam

# 2. Install all dependencies
source env_setup.sh
```

> **No conda?** You can also create a plain virtual environment:
> ```bash
> python -m venv prompt_cam_env
> source prompt_cam_env/bin/activate   # Windows: prompt_cam_env\Scripts\activate
> bash env_setup.sh
> ```

---

## 3. Downloading Checkpoints

Pre-trained checkpoints are hosted on Google Drive. Download the checkpoint for
the model/dataset pair you want to use and place it at:

```
checkpoints/{model}/{dataset}/model.pt
```

For example, the DINO + CUB checkpoint goes in `checkpoints/dino/cub/model.pt`.

Available checkpoints:

| Backbone | Dataset | Accuracy (top-1) | Download |
|---|---|---|---|
| dino | CUB-200 | 73.2% | [Google Drive](https://drive.google.com/drive/folders/1UmHdGx4OtWCQ1GhHCrBArQeeX14FqwyY?usp=sharing) |
| dino | Stanford Cars | 83.2% | [Google Drive](https://drive.google.com/drive/folders/1UmHdGx4OtWCQ1GhHCrBArQeeX14FqwyY?usp=sharing) |
| dino | Stanford Dogs | 81.1% | [Google Drive](https://drive.google.com/drive/folders/1UmHdGx4OtWCQ1GhHCrBArQeeX14FqwyY?usp=sharing) |
| dino | Oxford Pets | 91.3% | [Google Drive](https://drive.google.com/drive/folders/1UmHdGx4OtWCQ1GhHCrBArQeeX14FqwyY?usp=sharing) |
| dino | Birds-525 | 98.8% | [Google Drive](https://drive.google.com/drive/folders/1UmHdGx4OtWCQ1GhHCrBArQeeX14FqwyY?usp=sharing) |
| dinov2 | CUB-200 | 74.1% | [Google Drive](https://drive.google.com/drive/folders/1UmHdGx4OtWCQ1GhHCrBArQeeX14FqwyY?usp=sharing) |
| dinov2 | Stanford Dogs | 81.3% | [Google Drive](https://drive.google.com/drive/folders/1UmHdGx4OtWCQ1GhHCrBArQeeX14FqwyY?usp=sharing) |
| dinov2 | Oxford Pets | 92.7% | [Google Drive](https://drive.google.com/drive/folders/1UmHdGx4OtWCQ1GhHCrBArQeeX14FqwyY?usp=sharing) |

---

## 4. Running the Web App (Recommended for New Users)

After completing [Section 2](#2-environment-setup) and [Section 3](#3-downloading-checkpoints):

```bash
conda activate prompt_cam
python app.py
```

Open the URL displayed in your terminal (typically `http://localhost:7860`).

### Step-by-step walkthrough

1. **Select a Model / Dataset** from the drop-down (e.g., *DINO / CUB-200*).
2. **Upload a checkpoint** — click the file picker and select the `.pt` file you
   downloaded in Step 3.
3. **Choose an image** — either upload your own photo or select one of the
   built-in sample images (CUB birds) and click *Load sample →*.
4. **Set the target class index** — this is the class whose traits you want to
   visualize. For example, in CUB-200, class `97` is *Scott's Oriole*.
   - Class numbering starts at **0**.
   - The total number of classes per dataset is shown in the drop-down label.
5. **Set the number of traits** (attention heads) to highlight — 3 to 4 is a
   good starting point.
6. Click **▶ Run Visualization**.

The output panel shows the original image alongside the top-ranked trait
heatmaps overlaid on the image.

---

## 5. Running the Demo Notebook

If you prefer an interactive notebook environment:

```bash
conda activate prompt_cam
jupyter notebook demo.ipynb
```

The notebook walks through:
- Loading a model checkpoint
- Visualizing traits for a single image
- Comparing how different classes "see" the same image
- Trait manipulation examples

Edit the variables in the **first cell** to point to your chosen model, dataset
config, and checkpoint path.

---

## 6. Running Visualization from the Command Line

```bash
conda activate prompt_cam
python visualize.py \
    --config  ./experiment/config/prompt_cam/dino/cub/args.yaml \
    --checkpoint ./checkpoints/dino/cub/model.pt \
    --vis_cls 23
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--config` | — | Path to the YAML config file for the model/dataset |
| `--checkpoint` | — | Path to the trained `.pt` checkpoint |
| `--vis_cls` | `23` | Class index to visualize |
| `--top_traits` | `4` | Number of top attention heads to highlight |
| `--nmbr_samples` | `10` | Number of test images to process |
| `--vis_outdir` | `./visualization` | Output directory for saved images |

Output images are saved to `visualization/{model}/{dataset}/class_{N}/`.

---

## 7. Training a New Model

Training requires a GPU and the full dataset. See the [README Training
section](README.md#fire-training) for detailed instructions.

---

## 8. Extending the Codebase

### Adding a new dataset

1. Prepare your data as described in [Data Preparation](README.md#data-preparation).
2. Create a new dataset file in `data/dataset/` modelled on
   [`data/dataset/cub.py`](data/dataset/cub.py).
3. Register it in [`experiment/build_loader.py`](experiment/build_loader.py).
4. Create a config file at
   `experiment/config/prompt_cam/{model}/{dataset}/args.yaml` (copy and adapt
   an existing one).

### Adding a new backbone

1. Modify `get_base_model()` in [`experiment/build_model.py`](experiment/build_model.py).
2. Register the architecture in [`model/vision_transformer.py`](model/vision_transformer.py).
3. Add the new `--pretrained_weights` and `--model` choices to `setup_parser()`
   in [`main.py`](main.py).

---

## 9. Reporting Issues

Found a bug or have a question? Please [open an issue](https://github.com/Imageomics/Prompt_CAM/issues/new) and include:

- Your operating system and Python version (`python --version`)
- The exact command or steps you ran
- The full error message / stack trace (if any)
- Which model/dataset combination you were using

---

## 10. Code Style

- Python 3.10+
- Follow [PEP 8](https://peps.python.org/pep-0008/) for formatting.
- Use descriptive variable names; avoid single-letter names outside loop
  counters.
- Add docstrings to new public functions and classes.
- Do not commit large binary files (checkpoints, datasets) — these belong on
  external storage (Google Drive, Hugging Face Hub, etc.).
