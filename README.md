# PeleSENet: A Cutting-Edge Hybrid CNN for Enhanced Skin Cancer Detection with XAI and Adaptive Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation for **PeleSENet**, a novel lightweight hybrid Convolutional Neural Network (CNN) for skin cancer classification, as presented in the paper "PeleSENet: A Cutting-Edge Hybrid CNN for Enhanced Skin Cancer Detection with ΧΑΙ and Adaptive Learning".

PeleSENet is designed to be highly efficient and accurate, making it suitable for deployment in resource-constrained environments like mobile devices and clinical point-of-care systems. The framework also integrates **Explainable AI (XAI)** and an **Adaptive Learning** loop to ensure transparency, trust, and continuous improvement.

## 🌟 Key Features

*   **Hybrid Architecture:** Combines the strengths of **PeleeNet** (dense connectivity) and **SE-ResNet** (channel attention) to capture diverse and salient features.
*   **Lightweight & Efficient:** Achieves high performance with **fewer than 3.5 million parameters**, ensuring fast inference times and low computational overhead.
*   **High Accuracy:** Attains **89.70% accuracy** on the ISIC dataset, outperforming numerous non-pretrained models.
*   **Explainable AI (XAI):** Uses **Grad-CAM** to generate visual heatmaps, highlighting the image regions that are most influential in the model's predictions. This builds trust and aids in clinical validation.
*   **Adaptive Learning Framework:** A complete system that allows for continuous improvement. Low-confidence predictions are flagged for review by dermatologists, and their feedback is used to periodically retrain and enhance the model.

## 🧬 The Proposed Framework

Our system is more than just a model; it's a complete, end-to-end framework for reliable and evolving skin cancer detection.

1.  **Image Acquisition & Preprocessing:** Input images are captured and preprocessed with normalization and data augmentation.
2.  **PeleSENet Classification:** The image is fed into PeleSENet to get a classification (benign/malignant) and a confidence score.
3.  **Confidence-Based Decision Making:**
    *   **High-Confidence (>98%):** The result is delivered instantly with a Grad-CAM heatmap for explainability.
    *   **Low-Confidence (<98%):** The image and preliminary result are sent to a dermatologist dashboard for expert review.
4.  **Logging & Adaptive Retraining:** All results and expert feedback are logged. The model is periodically retrained on this growing dataset to adapt to new data patterns and improve its accuracy over time.


*Fig. 1: Malignant prediction with high confidence, showing the original image, the Grad-CAM heatmap, and the overlay.*


*Fig. 2: Benign prediction with high confidence.*

## 🏗️ Model Architecture

PeleSENet strategically integrates several key components:

*   **Dense Feature Blocks:** Inspired by PeleeNet, these blocks aggregate features from multiple layers to learn diverse representations.
*   **Light Residual Blocks with LightCBAM:** These blocks refine features using both channel and spatial attention (via the Convolutional Block Attention Module), helping the model focus on critical areas of the lesion.
*   **Transition Blocks:** These control model complexity and reduce dimensionality between stages, ensuring efficiency.


*Fig. 3: The overall architecture of PeleSENet.*

## 📊 Performance and Results

PeleSENet was trained and evaluated on a subset of the **ISIC 2018 dataset**. The model demonstrates a superior balance of accuracy and efficiency.

*   **Final Test Accuracy:** **89.70%**
*   **Precision (Malignant):** 91.1%
*   **Recall/Sensitivity (Malignant):** 85.7%


*Fig. 4: Accuracy and Loss curves over 35 epochs, showing stable training and good generalization.*


*Fig. 5: Confusion matrix on the test set, illustrating the model's performance in distinguishing between benign and malignant cases.*

## 🛠️ Setup and Installation


**Dataset:**
Download the ISIC dataset and organize it into `train` and `test` folders, with `benign` and `malignant` subdirectories inside each.

```
data/
├── train/
│   ├── benign/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── malignant/
│       ├── image_002.jpg
│       └── ...
└── test/
    ├── benign/
    │   └── ...
    └── malignant/
        └── ...
```


## 🙏 Acknowledgments

*   This work is based on the architectures of [PeleeNet](https://arxiv.org/abs/1804.06882) and [SE-ResNet](https://arxiv.org/abs/1709.01507).
*   The dataset used for this project is a subset of the [ISIC (International Skin Imaging Collaboration) archive](https://challenge.isic-archive.com/data/). We thank the organizers for making this valuable dataset publicly available.
