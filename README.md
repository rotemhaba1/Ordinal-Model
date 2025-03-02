# 📌 Preprocessing Stage - Ordinal Model Code

## 📂 Overview
This stage prepares raw EEG and respiratory cycle data for model training. It includes:
1. **Preprocessing** - Cleans and transforms raw data.
2. **Data Splitting** - Creates train-test and cross-validation (CV=5) splits.

---

## 🚀 How to Run the Preprocessing Pipeline

### 1️⃣ **Setup**
Make sure dependencies are installed:
```bash
pip install -r requirements.txt
```
Ensure your raw data is inside:
```
data/raw/
├── P_1/
│   ├── EEG_P_1.txt
│   ├── Measuring Time.xlsx
│   ├── challenge_test_report.xlsx
...
```

### 2️⃣ **Run the Preprocessing & Splitting**
Run the script to process all patients:
```bash
python src/preprocessing/save_processed_data.py
```

### 3️⃣ **Output Files**
Processed data and splits will be saved here:
```
data/processed/
├── fft_data_model_STFT_P_5.parquet
├── fft_data_model_STFT_P_6.parquet
...
data/splits/
├── split_train_test_P_5.parquet
├── split_train_test_P_6.parquet
...
```

---

## ✅ Steps in the Pipeline
1. **Preprocess Raw Data** (`prepare_data.py`)
   - Loads EEG & respiratory data.
   - Cleans and normalizes the data.
   - Saves output in `data/processed/`.
2. **Split Data** (`data_split.py`)
   - Performs **train-test split (80-20%)**.
   - Creates **5-fold cross-validation splits**.
   - Saves split files in `data/splits/`.
3. **Run Everything** (`save_processed_data.py`)
   - Runs preprocessing & splitting together.

---



