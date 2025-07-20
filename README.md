# 3D_safety_assessment
# Safety Assessment of 3D Generation Models in AR/VR Applications

[![DOI](https://img.shields.io/badge/DOI-XXXXXXX.XXXXXXX-blue?style=flat-square)](https://doi.org/XXXXXXX.XXXXXXX)

This repository provides the data and code for the paper *Safety Assessment of 3D Generation Models in AR/VR Applications*, accepted in ACM LAMPS '25.

**Paper Link**: [https://arxiv.org/pdf/YOUR_PAPER_ARXIV_LINK_HERE.pdf](https://arxiv.org/pdf/YOUR_PAPER_ARXIV_LINK_HERE.pdf)

## Introduction

This study conducts the first systematic safety audit of text-to-3D generation. We found that after sampling a state-of-the-art text-to-3D workflow using 6,137 prompts across five safety suites and double-blind annotating every output, **823 objects (13.41%) were flagged as unsafe**, confirming persistent risk in current generation workflows.

To address this challenge, we propose a **novel multimodal safety classifier** that fuses geometric and visual information for 3D content safety prediction.
* **Geometric Features**: A MeshCNN encoder distills shape features from simplified meshes.
* **Visual Features**: A ViT/Q-Former encoder captures complementary appearance information from multi-view renderings.
* **Classifier**: The resulting embeddings are fused and fed to a residual multilayer perceptron (Residual MLP) tuned for safety prediction using Focal Loss.

Our multimodal classifier achieves an accuracy of 0.8784 and an AUC-ROC of 0.8775, surpassing leading 2D image safety classification baselines, such as Unsafe Diffusion and ShieldGemma. Additionally, ablation studies confirm that multimodal fusion consistently outperforms either modality in isolation, underscoring the value of joint geometry-image reasoning for 3D safety.

### Architecture Overview

![Pipeline Overview](assets/pipeline_overview.png)
*Figure 1: Overview of the proposed multimodal 3D asset safety assessment pipeline.*

## Preparation

### 1. Dataset

The 3D safety dataset for this study was constructed through the following steps:
* **Prompt Collection**: Aggregated 6,137 prompts from i2p and UnsafeBench (including 4chan, Lexica, Template, MS COCO subsets).
* **3D Asset Generation**: Hunyuan3D v1.0 was used for end-to-end 3D asset generation from text prompts.
* **Data Annotation**: A rigorous double-blind annotation protocol was employed to classify 3D models as "safe" or "unsafe," with high inter-rater agreement (Fleiss' $x=0.77$).

Data should be organized such that `DATA_DIR` contains `train_list.txt` and `test_list.txt`, and each sample directory within `DATA_DIR` contains a `concat_embed.npy` file. `concat_embed.npy` represents the pre-computed 1024-dimensional combined embedding.

### 2. Environment Setup

**Prerequisites**:
* Python 3.x
* CUDA GPU (recommended)

**Installation**:
1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/qazwsx4008823823/3D_safety_assessment.git](https://github.com/qazwsx4008823823/3D_safety_assessment.git)
    cd 3D_safety_assessment
    ```

2.  **Install dependencies**:
    It is recommended to create and activate a Python virtual environment, then install all dependencies listed in `requirements.txt`:
    ```bash
    # (Optional) Create and activate a virtual environment
    # python3 -m venv venv
    # source venv/bin/activate

    pip install -r requirements.txt
    ```

    `requirements.txt` content:
    ```
    torch
    torchvision
    torchaudio
    numpy==1.26.4
    tqdm
    transformers
    accelerate
    safetensors
    Pillow
    requests
    trimesh
    open3d
    pyvista
    ```

## Usage

### Train the Multimodal Safety Classifier

The `train.py` script allows flexible configuration via command-line arguments.

```bash
# Run training with default parameters (DATA_DIR defaults to /dataset/outputs/path/, OUTPUT_DIR defaults to ./models/)
python train.py

# Example: Specify custom paths and hyperparameters
python train.py \
    --data_dir /path/to/your/processed_data \
    --output_dir /path/to/save/trained_models \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.0005 \
    --device cuda
