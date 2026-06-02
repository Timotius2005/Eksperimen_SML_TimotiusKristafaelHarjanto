# Eksperimen_SML_TimotiusKristafaelHarjanto

> **Criterion 1 — Advanced** | SMSML Dicoding  
> **Student:** Timotius Kristafael Harjanto  
> **Dataset:** Adult Income (UCI ML Repository) — 48,842 rows, Binary Classification

---

## Repository Structure

```
Eksperimen_SML_TimotiusKristafaelHarjanto/
│
├── .github/
│   └── workflows/
│       └── preprocessing.yml          ← GitHub Actions CI pipeline
│
├── dataset_raw/
│   └── adult.csv                      ← Auto-downloaded by pipeline
│
├── preprocessing/
│   ├── Eksperimen_TimotiusKristafaelHarjanto.ipynb   ← EDA Notebook
│   ├── automate_TimotiusKristafaelHarjanto.py        ← Automated pipeline
│   └── dataset_preprocessed/
│       ├── train.csv                  ← Auto-generated
│       ├── test.csv                   ← Auto-generated
│       └── feature_names.txt
│
└── README.md
```

---

## Dataset

**Adult Income (Census Income)** — UCI Machine Learning Repository  
URL: https://archive.ics.uci.edu/ml/datasets/adult  
Rows: 48,842 | Features: 14 raw → 20 engineered | Task: Binary Classification  
Target: `income` — whether income exceeds $50K/year (1) or not (0)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Timotius2005/Eksperimen_SML_TimotiusKristafaelHarjanto.git
cd Eksperimen_SML_TimotiusKristafaelHarjanto

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 4. Run the notebook (EDA)
cd preprocessing
jupyter notebook Eksperimen_TimotiusKristafaelHarjanto.ipynb

# 5. Run the automated pipeline
python automate_TimotiusKristafaelHarjanto.py
```

---

## Automated Preprocessing Pipeline

`preprocessing/automate_TimotiusKristafaelHarjanto.py` implements:

| Function | Description |
|---|---|
| `load_data()` | Download from UCI and cache locally |
| `clean_data()` | Drop NaN rows, remove duplicates, cap outliers (3×IQR) |
| `feature_engineering()` | Age groups, capital net, education flag, hours category, married flag |
| `encode_features()` | LabelEncoder for all categoricals; binarise target |
| `scale_features()` | StandardScaler + stratified 80/20 train-test split |
| `save_processed_dataset()` | Persist train.csv, test.csv, feature_names.txt |

---

## GitHub Actions CI

`.github/workflows/preprocessing.yml` automatically:

1. Triggers on `push` (when preprocessing files change) or `workflow_dispatch`
2. Installs dependencies
3. Runs `automate_TimotiusKristafaelHarjanto.py`
4. Commits the updated `dataset_preprocessed/` back to the repository
5. Uploads dataset as a downloadable GitHub Artifact

---

## EDA Notebook Coverage

The notebook `Eksperimen_TimotiusKristafaelHarjanto.ipynb` covers:

1. Data Loading & Shape
2. Data Inspection (dtypes, value counts)
3. Missing Value Analysis (visualised with heatmap)
4. Duplicate Analysis
5. Outlier Analysis (boxplots, IQR method)
6. Exploratory Data Analysis (distributions, correlations)
7. Feature Engineering (7 new features)
8. Encoding (LabelEncoder)
9. Scaling (StandardScaler)
10. Train/Test Split (stratified, 80/20)

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Dataset download fails | Check internet connection; UCI URL may be temporarily unavailable |
| Module not found | Run `pip install -r requirements.txt` |
| GitHub Actions fails | Ensure workflow file has correct Python version |
| Empty processed dataset | Delete `dataset_raw/adult.csv` to force re-download |
