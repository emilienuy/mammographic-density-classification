# Mammographic Breast Density Classification

Breast density classification from screening mammograms using deep learning, with subgroup fairness analysis and embedding-space model inspection.

Built as part of the **Machine Learning for Imaging** course at Imperial College London (MSc AI, 2025–26).

📓 [View notebook on Colab](https://colab.research.google.com/drive/1sTlixwFxJy_WC3TfSQESeSNIPAiMVxem?usp=sharing) · 📄 [Read the report](report.pdf)

---

## Task

Predict BI-RADS breast density category from mammographic images:

| Class | Description |
|-------|-------------|
| A | Almost entirely fatty |
| B | Scattered fibroglandular density |
| C | Heterogeneously dense |
| D | Extremely dense |

---

## Approach

**Data augmentation**: Separate photometric and geometric pipelines designed to simulate realistic acquisition variability (scanner differences, positioning shifts, exposure variation) without distorting density-relevant tissue structure.

**Model selection**: Compared MLP baseline, EfficientNet-B0, ResNet-18, and DenseNet-121 (all CNNs with ImageNet pretraining). DenseNet-121 selected based on highest validation AUROC and more stable training dynamics.

**Subgroup analysis**: Evaluated performance across age, implant status, imaging view, scanner ID, and laterality to assess robustness and fairness.

**Model inspection**: Extracted penultimate-layer embeddings and analysed the learned representation with PCA and t-SNE, comparing against raw pixel PCA as a baseline.

---

## Results

| Model | Val AUROC |
|-------|-----------|
| MLP baseline | 0.818 |
| EfficientNet-B0 | 0.872 |
| ResNet-18 | 0.919 |
| **DenseNet-121** | **0.929** |

**Test AUROC (DenseNet-121): 0.9143**

### Subgroup findings

- **Age**: Performance increases with age (0.865 for <40 → 0.931 for 70+), likely reflecting lower mammographic density variability in older patients
- **Implants**: Largest performance gap — macro-AUC drops from 0.921 (no implant) to 0.784 (implant), suggesting limited generalisation to implant cases
- **Scanner**: Moderate variation across 5 scanners (0.894–0.941), with no dominant scanner effect in the embedding space
- **Laterality / view**: Negligible differences, confirming robustness to projection angle and side

### Embedding analysis

PCA of learned DenseNet-121 embeddings shows a continuous, ordered structure along the first principal component (A→D), consistent with the ordinal nature of density grading. Misclassifications concentrate in overlap regions between adjacent classes, indicating borderline cases rather than systematic errors.

---

## Stack

PyTorch · torchvision · PyTorch Lightning · scikit-learn · Matplotlib

---

## Authors

Emilie Nuyttens & George Stamatopoulos at Imperial College London
