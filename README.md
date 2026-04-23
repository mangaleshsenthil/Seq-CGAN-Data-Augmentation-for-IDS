# Seq-CGAN-IDS: Sequential Conditional GAN-Based Data Augmentation for Imbalanced Intrusion Detection

## Overview

This repository implements a Sequential Conditional Generative Adversarial Network (Seq-CGAN) framework for addressing extreme class imbalance in intrusion detection systems. The approach focuses on generating realistic temporal sequences for rare attack classes and improving detection performance using sequence-based deep learning models.

## Key Features

* Sequential data augmentation using class-conditional GAN
* Sliding window-based temporal sequence construction
* Rare-class focused synthetic data generation
* BiLSTM-based intrusion detection model
* Imbalance handling using focal loss and weighted sampling
* Evaluation using macro-precision, macro-recall, and macro-F1
* Five-fold stratified cross-validation
* t-SNE-based validation of synthetic data fidelity

## Dataset
This project uses the UNSW-NB15 dataset, which contains realistic network traffic with multiple attack categories and inherent class imbalance.

Official source:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

For convenience, a preprocessed version is also available on Kaggle:
https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Download the dataset and place the CSV file in the project directory before running the notebook.

Steps applied:

* Removal of non-generalizable identifiers
* One-hot encoding of categorical features
* Yeo–Johnson Power Transformation for normalization
* Conversion to temporal sequences using sliding window (T = 10)

## Methodology

The pipeline consists of four main stages:

1. Data Preprocessing
   Raw tabular data is cleaned, encoded, normalized, and transformed into temporal sequences.

2. Seq-CGAN Augmentation
   A conditional LSTM-based generator produces synthetic sequences for rare classes, while a BiLSTM discriminator evaluates sequence realism.

3. Model Training
   Sequence-based classifiers (primarily BiLSTM) are trained on augmented data using focal loss and weighted sampling.

4. Evaluation
   Performance is measured using macro-averaged metrics and cross-validation to ensure robustness.

## Model Architecture

### Generator

* Two-layer LSTM
* Input: noise vector + class embedding
* Output: synthetic sequence of shape (T, F)

### Discriminator

* Bidirectional LSTM
* Input: sequence + class embedding
* Output: real/fake classification score

### Classifier

* BiLSTM-based intrusion detection model
* Fully connected classification head
* Trained using focal loss

## Training Details

* Optimizer: Adam
* Learning rate: 1e-4 (GAN), 1e-3 (classifier)
* Batch size: 64
* Epochs: 10 (classifier), 150 (GAN)
* Label smoothing applied to discriminator
* WeightedRandomSampler for class balancing

## Evaluation Metrics

* Macro Precision
* Macro Recall
* Macro F1 Score
* Confusion Matrix
* Per-class Recall

Macro metrics are prioritized to ensure fair evaluation under class imbalance.

## Results Summary

* Seq-CGAN improves macro-F1 compared to non-augmented baselines
* BiLSTM achieves the best balance across classes
* Rare-class detection (Worms, Shellcode) improves significantly
* Cross-validation results demonstrate stable performance

## Ablation Study

The following configurations are evaluated:

* Baseline (Cross-Entropy)
* Focal Loss
* Weighted Sampling
* Focal + Weighted Sampling
* Seq-CGAN + Focal + Weighted Sampling

Results show that Seq-CGAN provides the most balanced improvement across precision, recall, and F1.

## Dependencies

* Python 3.8+
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

## File Structure

* notebook.ipynb: Complete pipeline implementation
* README.md: Project documentation

## Notes
* Only final models and configurations are retained for clarity
