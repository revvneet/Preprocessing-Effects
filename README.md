# BMI Classification - Preprocessing Comparison Project

A machine learning project demonstrating various data preprocessing techniques for BMI classification using Logistic Regression.

## Project Overview

This project explores different preprocessing approaches to improve model performance on a BMI classification task:
- Baseline model
- Handling missing values (imputation vs dropping)
- Class balancing with SMOTE
- Feature scaling
- Hyperparameter tuning (max_iter)

## Dataset

- **Features**: Age, Height, Weight, Bmi
- **Target**: BmiClass (Normal Weight, Overweight, Obese Class 1, Obese Class 2, Obese Class 3, Underweight)

## Project Structure

```
.
├── data/
│   ├── bmi.csv              # Canonical dataset
│   └── bmi_missing.csv      # Dataset with missing values injected
├── outputs/
│   ├── baseline/           # Baseline model results
│   ├── missing_values_imputed/    # KNN imputation results
│   ├── missing_values_dropped/     # Drop rows with missing values
│   ├── smote_scaling/   # SMOTE + StandardScaler results
│   ├── max_iter/       # Increased max_iter results
│   └── recall_fix/    # Class weight balanced results
├── experiments/
│   ├── baseline/TrainModel.py
│   ├── missing_values_imputed/TrainModel.py
│   ├── missing_values_dropped/TrainModel.py
│   ├── smote_scaling/TrainModel.py
│   ├── max_iter/TrainModel.py
│   └── recall_fix/TrainModel.py
├── docs/
│   ├── DICTIONARY.md
│   ├── quick_start.md
│   └── results_index.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the Experiments

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run a Specific Variant

```bash
# Baseline
python experiments/baseline/TrainModel.py

# With missing value imputation
python experiments/missing_values_imputed/TrainModel.py

# With SMOTE and scaling
python experiments/smote_scaling/TrainModel.py

# Max iter increase
python experiments/max_iter/TrainModel.py
```

## Results

Each experiment produces:
- Trained model (.pkl)
- Classification report
- Visualization PNGs

See `docs/results_index.md` for a summary of all results.

## Dependencies

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- joblib