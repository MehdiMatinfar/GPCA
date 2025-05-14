# 🧠 Brain Graph Analysis Pipeline

This project processes fMRI NIfTI data into functional connectivity graphs, extracts temporal windows, and prepares them for training graph neural networks. It includes feature extraction using atlases, Kernel PCA dimensionality reduction, synthetic data augmentation via `condica`, and time-series graph construction.

---

## 🚀 How to Run the Project

### 🧰 Prerequisites

Make sure the following are installed:

- Python 3.8+
- pip
- `conda` (optional, but recommended)

### 📦 Required Python Packages

Install dependencies using:

```bash
pip install -r requirements.txt



Or, with Conda:

```bash

conda create -n brain-gcn-env python=3.8
conda activate brain-gcn-env
pip install -r requirements.txt




📁 Directory Structure
Ensure your project directory looks like this:


[Link of dataset]([https://link-url-here.org](https://zenodo.org/records/5123331))


project/
│
├── data_set/
│   ├── taowu_patients.tsv
│   ├── data_set_patients.tsv
│   └── *.nii.gz (ending with 'old.nii.gz')
│
├── mask/
│   ├── hcp_mask.nii.gz
│   └── difumo_atlases/ (created automatically if missing)
│
├── condica/
│   ├── main.py
│   ├── utils.py
│   └── ...
│
├── gcn_windows_dataset_test.py
├── main_script.py
├── requirements.txt
└── README.md


🏃‍♂️ Running the Pipeline

```bash
python main.py


This script will:

Find all NIfTI files ending with old.nii.gz.

Apply brain masking and feature extraction using NiftiMasker.

Reduce features with Kernel PCA.

Use the DiFuMo atlas to generate more features.

Use the condica module to generate synthetic samples.

Compute pairwise correlation matrices.

Construct time windowed graphs and save them.

Load the dataset using TimeWindowsDataset.

📌 Notes
Internet access may be required the first time to download the DiFuMo atlas.

The code assumes all NIfTI files are spatially normalized and preprocessed.

condica must be a valid local module with main.py and utils.py.

