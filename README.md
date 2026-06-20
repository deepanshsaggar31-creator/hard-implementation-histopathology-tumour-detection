# Histopathology Tumor Detection and Segmentation Pipeline

This repository contains the Jupyter Notebook and standalone Python scripts for histopathology tumor detection and segmentation using **ResNet-UNet** and **Attention-UNet** models.

## 📁 Repository Contents

*   **`histopathology_tumour_detection.ipynb`**: The original Jupyter Notebook containing data loading (for Kumar dataset), WSI slide simulation, U-Net and Attention U-Net training, FROC curve plotting, and stain normalization ablation studies.
*   **`utils_segmentation.py`**: Custom helper functions including `otsu_threshold`, `otsu_tissue_mask`, `normalize_stain_macenko` stain normalization, `WSI_Simulator`, and `TissuePatchDataset` dataset classes.
*   **`unet_model.py`**: Model architecture classes for `DoubleConv`, `AttentionBlock`, `ResNetUNet` (standard U-Net), `AttentionUNet` (Attention U-Net), and the combined `DiceBCELoss` function.
*   **`run_notebook_pipeline.py`**: A combined, standalone Python script representing all cells of the notebook (loading datasets, training models, and generating all evaluation reports).

---

## 🛠️ Requirements & Installation

Install the required packages to run the notebook and scripts:
```bash
pip install torch torchvision numpy matplotlib opencv-python pillow scikit-learn
```

---

## 🔬 How to Run

1.  **Jupyter Notebook**: Open `histopathology_tumour_detection.ipynb` in any Jupyter environment (or Google Colab / Kaggle).
2.  **Standalone Script**: Run the pipeline locally via terminal:
    ```bash
    python run_notebook_pipeline.py
    ```
