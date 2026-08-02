# IPA-CycleGAN — Face Aging with Identity Preservation

A deep learning project that performs **realistic face aging while preserving identity** using a custom CycleGAN-based architecture. The model learns age-related facial transformations across **8 age groups** without requiring paired before/after images of the same person.

Built with **PyTorch** • Trained on **UTKFace** • Evaluated on **AgeDB-30**

---

## Overview

The goal of this project is to generate an aged version of a person's face while maintaining their identity.

Unlike supervised image translation methods, this model is trained using **unpaired images**, allowing it to learn facial aging patterns without requiring photographs of the same individual at different ages.

### Input
A face image

### Output
The same person's face translated into a selected target age group.

Example:

```
20s  →  60s
```

The model captures both **structural** and **textural** aging changes while minimizing identity loss.

---

# Architecture

The proposed model combines multiple deep learning components for realistic age progression.

| Component | Purpose |
|-----------|---------|
| **FaceEncoder (ResNet50)** | Separates facial identity from age-related features |
| **AgeEmbedder (MLP)** | Learns an embedding representing the target age group |
| **U-Net Generator + AdaIN** | Generates the aged face using adaptive style injection |
| **Attention Gates** | Focuses generation on age-sensitive facial regions |
| **ArcFace (Frozen ResNet50)** | Computes identity preservation loss |
| **Multi-Scale Discriminator + Spectral Normalization** | Improves adversarial training stability |
| **CycleGAN Framework** | Enables learning from unpaired datasets |
| **VGG16 Perceptual Network** | Preserves high-frequency texture and visual quality |

---

# Training Objective

The final objective combines multiple losses:

```
Total Loss =
    Adversarial Loss (WGAN-GP)
  + Cycle Consistency Loss
  + Identity Loss (ArcFace)
  + Age Classification Loss
  + Perceptual Loss (VGG16)
  + Feature Matching Loss
```

Each loss contributes to a different aspect of image generation:

- **Adversarial Loss** → Realistic image generation
- **Cycle Loss** → Preserves facial structure
- **Identity Loss** → Maintains person identity
- **Age Classification Loss** → Ensures correct age translation
- **Perceptual Loss** → Preserves image quality
- **Feature Matching Loss** → Stabilizes GAN training

---

# Datasets

## 1. UTKFace

Used for model training.

- Approximately **14,500 images**
- Balanced across **8 age groups**
- Images deduplicated using **Perceptual Hashing (pHash)**

### Age Groups

| Label | Age Range |
|-------|-----------|
| 0 | 0–10 |
| 1 | 11–20 |
| 2 | 21–30 |
| 3 | 31–40 |
| 4 | 41–50 |
| 5 | 51–60 |
| 6 | 61–70 |
| 7 | 71+ |

---

## 2. AgeDB-30

Used exclusively for evaluation.

- **500 evaluation images**
- Used for identity similarity and image quality metrics

---

# Results

| Metric | Score |
|---------|------:|
| **Cosine Identity Similarity (CSIM)** | **~0.88** |
| **Fréchet Inception Distance (FID)** | **~51** |

### Interpretation

- **CSIM ≈ 0.88**
  - Indicates that facial identity is well preserved after age translation.

- **FID ≈ 51**
  - Demonstrates reasonable image quality considering the limited training resources.

---

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install facenet-pytorch timm tqdm imagehash
pip install torch-fidelity Pillow==10.2.0
```

---

# Usage

Run the notebooks in the following order:

```
1. PREPROCESSING.ipynb
          ↓
2. TRAINING.ipynb
          ↓
3. TESTING.ipynb
```

Google Colab with GPU runtime is recommended.

---

# Project Structure

```
IPA-CycleGAN/
│
├── PREPROCESSING.ipynb
├── TRAINING.ipynb
├── TESTING.ipynb
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   ├── encoder.py
│   └── losses.py
│
├── utils/
├── datasets/
├── results/
├── README.md
└── requirements.txt
```

*(Modify the structure according to your repository.)*

---

# Limitations

This project was completed as part of a semester project under limited computational resources.

- Training dataset was reduced due to GPU and runtime constraints.
- Training was limited to approximately **70–100 epochs** on Google Colab.
- Images were generated at **128 × 128 resolution**, which can appear soft when enlarged.
- The model predicts **8 discrete age groups** rather than continuous ages.
- Trained checkpoints are not included because they exceed GitHub's file size limits.
- Since UTKFace contains no paired age progression images, training relies on an **unpaired CycleGAN framework**.

---

# Future Work

Potential improvements include:

- Train on the complete UTKFace dataset
- Increase image resolution to **256×256** or **512×512**
- Replace age groups with continuous age conditioning
- Improve FID using longer training and larger GPUs
- Add StyleGAN-based decoder
- Release pretrained checkpoints via Google Drive or Hugging Face

---

# Technologies Used

- PyTorch
- CycleGAN
- ResNet50
- ArcFace
- U-Net
- AdaIN
- Attention Gates
- VGG16
- WGAN-GP
- Spectral Normalization
- Google Colab
- UTKFace
- AgeDB-30

---

# References

- CycleGAN
- ArcFace
- UTKFace Dataset
- AgeDB-30 Dataset
- WGAN-GP
- VGG Perceptual Loss

---
