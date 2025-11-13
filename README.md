# Computer Vision Assignment – Traffic Sign Dataset (MSA 8770)

## Overview
This project processes the **Mapillary Traffic Sign Dataset (MTSD v2 fully annotated)** as part of the MSA 8770 course.  
The work includes:

- Parsing 41,909 JSON annotation files  
- Extracting image–annotation mappings  
- Identifying top-5 most frequent traffic sign categories  
- Creating a balanced subset of ~500 images  
- Splitting data into **train / val / test**  
- Visual sanity checks  
- Automatic cropping of labeled objects  
- Exporting metrics and subset indices

All work was completed on **GSU ARC JupyterHub**, and the full code is contained in the `notebooks/` directory.

---

## Repository Structure

```
computer-vision-assignment/
│
├── notebooks/
│   └── 01_data_preparation.ipynb        # Full preprocessing pipeline
│
├── results/
│   └── metrics/
│       ├── subset_index.csv             # Final subset CSV
│       └── (other generated files)
│
└── README.md                            # Project description
```
## ⚙️ Workflow

* **Phase 1A: Data Preparation (`01_data_preparation.ipynb`)**
    * Parsed 95k+ JSON annotations and validated them against 25k+ existing images.
    * Created `annotations_master.csv` with all valid bounding boxes.
    * Identified the top 5 most frequent classes (excluding "other-sign").
    * Created a balanced `train`/`val`/`test` subset (`subset_split.csv`) for these classes.
    * Generated exploratory analysis plots (class balance, sign size distribution).

* **Phase 1B & 1C: Preprocessing & Region Proposal (`02_preprocessing_region_proposal.ipynb`)**
    * **1B:** Developed a preprocessing pipeline (CLAHE, Bilateral Filter) and saved 128x128 standardized sign crops.
    * **1B:** Generated color space analysis visualizations.
    * **1C:** Implemented a classical region proposal pipeline (Color + Shape detection + NMS).
    * **1C:** Evaluated proposals against ground truth using IoU and saved metrics.

* **Phase 2: Classical ML & Deep Learning (`03_classical_ml.ipynb` & `04_deep_learning.ipynb`)**
    * Extracted HOG + HSV Histogram features for classical models.
    * Trained and evaluated a **LinearSVC (SVM)**.
    * Trained and evaluated a **Random Forest** (ensemble method).
    * Built, trained, and evaluated a **SimpleCNN** from scratch using PyTorch.

* **Phase 3: Analysis (`05_analysis.ipynb`)**
    * Aggregated metrics from all 3 models into a final comparison table.
    * Generated the final model comparison bar chart.
    * Generated and displayed failure analysis plots for the best model (SVM) and the worst model (CNN).
    * Wrote final conclusions and recommendations.

## 📊 Results Summary

The classical SVM with engineered features was the clear winner, while the simple CNN failed to train effectively on the small, imbalanced dataset.

| Model | Test Accuracy | Validation Accuracy |
| :--- | :---: | :---: |
| **SVM (HOG+HSV)** | **0.956** | **0.933** |
| Random Forest (HOG+HSV) | 0.882 | 0.925 |
| SimpleCNN | 0.504 | 0.529 |

**Insight:** Classical SVM with engineered features (HOG+HSV) significantly outperformed the simple CNN, likely due to the small/imbalanced dataset being insufficient for the CNN to learn meaningful features from scratch.

## 🧠 Key Learnings

* The importance of robust data cleaning (e.g., matching all annotations to existing images).
* The power of classical, engineered features (HOG+HSV) for well-defined classification tasks.
* The difficulty of training a CNN from scratch on a small, imbalanced dataset.
* The value of failure analysis to understand *why* a model fails (e.g., SVM confusing `stop` and `no-entry` signs).

## 🛠️ Environment Setup

This project was built on the GSU ARC cluster. To run, use an Anaconda environment with the following packages:


conda install -c conda-forge jupyterlab pandas numpy matplotlib opencv scikit-learn seaborn joblib pytorch torchvision

## What the Code Does

### **1. Loads and parses MTSD annotations**
- Reads all `.json` files from  
  `/data/project/MSA8395/mapillary_traffic_sign_dataset/mtsd_v2_fully_annotated/annotations/`
- Extracts:
  - image ID  
  - bounding boxes  
  - class labels  

### **2. Identifies top-5 most frequent traffic sign classes**
Using `Counter`, the notebook finds the five most common categories.

### **3. Creates a balanced subset**
For each top class:
- randomly selects up to 100 images  
- merges into a 498-image subset  
- saves to `subset_index.csv`

### **4. Splits into Train / Validation / Test**
- 70% train  
- 15% validation  
- 15% test  

### **5. Visual sanity check**
Random image displayed along with its labels.

### **6. Crops individual sign objects**
- Loads image  
- Applies padding  
- Resizes to 128×128 while preserving aspect ratio  
- Saves each crop into folders:

results/crops/train/
results/crops/val/
results/crops/test/


---

## How to Run (ARC JupyterHub)

1. Start JupyterLab on ARC  
2. Open the notebook:

notebooks/01_data_preparation.ipynb

3. Run all cells top-to-bottom  
4. Outputs (CSV + crops) appear automatically under `results/`

---

## Requirements
- Python 3  
- Pandas  
- PIL (Pillow)  
- tqdm  
- matplotlib  
- GSU ARC-accessible dataset paths  

Everything is already installed on ARC.

---

## Author
**Abhay Prabhakar**  
MSA 8770 – Georgia State University  
Fall 2025

---

## Project Status
Completed.  
Final version submitted and pushed to GitLab.

