# 🧠 Glymphatic MRI Analysis Pipeline

This repository provides a Python-based MRI processing pipeline for analyzing dynamic contrast-enhanced MRI images to quantify glymphatic function. The project evaluates how Low-Intensity Focused Ultrasound (LIFU) can accelerate brain waste clearance.

## 🚀 Features
- Load and preprocess 4D dynamic MRI data
- Interactive ROI selection (glymphatic vs reference regions)
- Signal normalization and visualization
- Automated metric extraction: time-to-peak and peak-to-baseline
- Group-level statistical comparison (Wilcoxon Rank-Sum)
- Full automation of CSV output and plots

<img src="https://github.com/user-attachments/assets/dcba24fb-4a32-49dc-9ba9-1e99ff42bcb4" width="50%" height="50%">
<img src="https://github.com/user-attachments/assets/dac06769-12fc-437c-968f-11a951968319" width="15%" height="15%">

## 📁 Files Overview

| File                       | Purpose | Code Flow Explanation |
|----------------------------|---------|-----------------------|
| **`processing_code_v3_2.py`** | **Main Pipeline** | 1. Load and preprocess 4D MRI data (DICOM series).<br>2. Resample data and select target slice interactively.<br>3. Denoise images with Gaussian smoothing.<br>4. Interactively select glymphatic and muscle ROIs (`ROIselection.py`).<br>5. Normalize brain ROI signals using muscle ROI signals.<br>6. Compute signal intensity over time.<br>7. Calculate glymphatic metrics (`Metrics.py`).<br>8. Save signal results and metrics to CSV.|
| **`ROIselection.py`**         | Interactive ROI Selection | GUI-based tool for manually drawing Regions-of-Interest on MRI slices. Outputs coordinates and binary masks used for downstream analyses. |
| **`Metrics.py`**              | Metrics & Statistical Analysis | Computes glymphatic metrics (time-to-peak, peak-to-baseline) and performs statistical tests (Wilcoxon Rank-Sum) between groups (control vs. sweeping). |
| **`Metrics_calculate.py`**    | Threshold & Variance Analysis | Evaluates glymphatic influx/efflux metrics and calculates signal variance over time, performing statistical analyses. |
| **`Kim_Liana_report.pdf`**    | Detailed Research Report | Provides the theoretical background, study purpose, detailed methods, statistical interpretation, and conclusions from the conducted experiments. |
| **`requirements.txt`**        | Python Package Requirements | Lists all Python libraries required to run the pipeline. |

## 🖥️ Overall Workflow Diagram
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                            MRI Data Acquisition                             │
    │ (4D Dynamic MRI Images: Control vs Sweeping, multiple time points over 24h) │
    └─────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    Data Preprocessing (`processing_code_v3_2.py`)           │
    │    - Load and Resample Images                                               │
    │    - Interactive Slice Selection                                            │
    │    - Gaussian Noise Reduction                                               │
    └─────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                 Interactive ROI Selection (`ROIselection.py`)               │
    │    - Glymphatic ROI                                                         │
    │    - Muscle ROI (Normalization Region)                                      │
    └─────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                 Signal Extraction and Normalization                         │
    │    - Glymphatic ROI Signal Intensity Calculation                            │
    │    - Normalization using Muscle ROI Intensity                               │
    │    - Generate Time-course Signal Plots                                      │
    └─────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    Glymphatic Metrics (`Metrics.py`)                        │
    │    - Calculate Time-to-Peak & Peak-to-Baseline Metrics                      │
    │    - Perform Statistical Analysis (Wilcoxon Rank-Sum Test)                  │
    └─────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                Calculate and Statistical Analysis (`Metrics_calculate.py`)  │
    │    - Save metrics results and signals to CSV                                │
    │    - Calculate Glymphatic metrics under 'Metrics.py' analysis               │
    │    - Statistical analysis of metrics group comparison and signal variance   │
    └─────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                          Visualization & Reporting                          │
    │    - Metrics and signal intensity plots                                     │
    │    - Statistical Results Interpretation                                     │
    │    - Results saved in CSV and PNG                                           │
    └─────────────────────────────────────────────────────────────────────────────┘





## 🧪 Dependencies

Install the required packages:
```bash
pip install -r requirements.txt
```

## 📊 Run the Pipeline

1. Modify paths to your local DICOM folders inside `processing_code_v3_2.py`
2. Execute:
```bash
python processing_code_v3_2.py
```
3. Follow the instructions in the terminal to select ROI and run analysis.

## 📘 Report

For detailed methodology and results interpretation, refer to `Kim_Liana_report.pdf`.

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

Developed by Liana Kim as part of research on ultrasound-enhanced glymphatic clearance.
